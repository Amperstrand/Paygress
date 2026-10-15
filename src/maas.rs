// MAAS compute backend (Amperstrand spike, conwrt issue #194): workloads
// are MAAS machines composed on a VM host — each tenant gets a whole VM
// with its own kernel, so the isolation story matches the KVM backend
// (`IsolationLevel::DedicatedHost`).
//
// `create_container` implements the deploy-without-commissioning path:
// compose → mark-broken → fresh-MAC interface attach → mark-fixed (Ready,
// no commissioning) → deploy. A fresh compose on a VM host otherwise
// commission (~8 min, outside the provider's 300s backend-call budget);
// the mark-broken/attach/mark-fixed cycle lands the machine in Ready
// without running commissioning, leaving deploy as the only long step.
// The method resolves once deploy is ACCEPTED (state `Deploying`) — the
// orchestrator learns the address later via `get_container_ip`, mirroring
// how the KVM backend returns while the VM is still booting.
//
// MAAS credentials never live in the repo: url / api-key / vm-host come
// from the provider config, with env fallbacks PAYGRESS_MAAS_URL,
// PAYGRESS_MAAS_API_KEY ("consumer:token:secret"), PAYGRESS_MAAS_VM_HOST.
// The api key is rendered as an OAuth1 header; PLAINTEXT signatures are
// pinned for TLS-terminated region access, HMAC-SHA1 lands with live
// validation if the region requires it.
//
// The raw client (`MaasClient`) is one HTTP shape per region call so the
// wiremock tests can pin every verb/param; field names the region reports
// for VM-host capacity are the spike's contract and get re-pinned against
// a live region.

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use rand::Rng;
use reqwest::{Client, Url};
use serde::Deserialize;

use crate::compute::{
    container_name, ComputeBackend, ContainerConfig, ContainerStatus, NodeStatus,
};

/// Poll cadence for the mark-fixed (→ Ready) transition: a state flip the
/// region performs between two API calls, not a commissioning run.
const READY_POLL_INTERVAL: Duration = Duration::from_secs(2);
const READY_POLL_ATTEMPTS: u32 = 15;

/// One machine as the region reports it (the fields this backend needs).
#[derive(Debug, Clone, Deserialize)]
pub struct MaasMachine {
    pub system_id: String,
    pub hostname: String,
    pub status_name: String,
    #[serde(default)]
    pub ip_addresses: Vec<String>,
}

impl MaasMachine {
    fn is_ready(&self) -> bool {
        self.status_name == "Ready"
    }

    fn is_deploying_or_deployed(&self) -> bool {
        self.status_name == "Deploying" || self.status_name == "Deployed"
    }
}

/// Capacity shape of a VM host (the spike's contract; re-pin field names
/// against a live region during validation). Storage fields exist on the
/// wire but are deliberately not parsed: pooled VM-host storage is not a
/// node disk reading, and pretending otherwise would arm the spawn-time
/// headroom gate with a number it cannot trust.
#[derive(Debug, Clone, Deserialize)]
struct MaasVmHost {
    #[serde(default)]
    cpu_total: f64,
    #[serde(default)]
    cpu_used: f64,
    #[serde(default)]
    memory_total: u64,
    #[serde(default)]
    memory_used: u64,
}

/// Minimal region client: base URL + OAuth header + one method per API
/// shape. No state, clone-safe (pooled reqwest client).
#[derive(Debug, Clone)]
pub struct MaasClient {
    http: Client,
    base: Url,
    auth: Option<String>,
}

impl MaasClient {
    pub fn new(base_url: &str, api_key: &str) -> Result<Self> {
        let mut base = Url::parse(base_url).context("parse MAAS base URL")?;
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path()));
        }
        let auth = oauth1_plaintext(api_key);
        Ok(Self {
            http: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .context("build MAAS http client")?,
            base,
            auth: Some(auth),
        })
    }

    fn machine_url(&self, system_id: &str) -> Url {
        self.base
            .join(&format!("machines/{system_id}/"))
            .expect("valid base")
    }

    fn machines_url(&self) -> Url {
        self.base.join("machines/").expect("valid base")
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> Result<reqwest::Response> {
        let req = match &self.auth {
            Some(v) => req.header("Authorization", v),
            None => req,
        };
        let resp = req.send().await.context("MAAS http call")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!(
                "MAAS API error {status}: {}",
                body.chars().take(300).collect::<String>()
            );
        }
        Ok(resp)
    }

    async fn op(&self, system_id: &str, op: &str) -> Result<()> {
        self.send(
            self.http
                .post(self.machine_url(system_id))
                .query(&[("op", op)]),
        )
        .await
        .map(|_| ())
    }

    pub async fn machines(&self) -> Result<Vec<MaasMachine>> {
        self.send(self.http.get(self.machines_url()))
            .await?
            .json()
            .await
            .context("decode machines list")
    }

    /// `Err(None)`-style 404 signalling: `Ok(None)` = no such machine.
    pub async fn machine(&self, system_id: &str) -> Result<Option<MaasMachine>> {
        let resp = self
            .send(self.http.get(self.machine_url(system_id)))
            .await
            .map_err(|e| {
                // A 404 is an answer, not a failure: the sweep deletes
                // unconditionally and a re-run must tolerate a reaped id.
                if e.to_string().contains("404") {
                    anyhow!("not found")
                } else {
                    e
                }
            });
        match resp {
            Ok(resp) => Ok(Some(resp.json().await.context("decode machine")?)),
            Err(e) if e.to_string().contains("404") => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn compose(
        &self,
        vm_host_id: u32,
        hostname: &str,
        cores: u32,
        memory_mb: u32,
        storage_gb: u32,
    ) -> Result<MaasMachine> {
        let url = self
            .base
            .join(&format!("vm-hosts/{vm_host_id}/compose/"))
            .expect("valid base");
        let resp = self
            .send(self.http.post(url).form(&[
                ("hostname", hostname),
                ("cores", &cores.to_string()),
                ("memory", &memory_mb.to_string()),
                ("disks", &format!("{storage_gb}:{hostname}")),
            ]))
            .await?;
        resp.json().await.context("decode composed machine")
    }

    /// Attach a fresh, locally-administered MAC to a machine. The
    /// mark-broken → attach → mark-fixed cycle is what makes the region
    /// treat the machine as newly-seen hardware and skip commissioning.
    async fn attach_fresh_interface(&self, system_id: &str) -> Result<()> {
        let mac = fresh_local_mac();
        let url = self
            .base
            .join(&format!("machines/{system_id}/interfaces/"))
            .expect("valid base");
        self.send(
            self.http
                .post(url)
                .form(&[("mac_address", mac.as_str()), ("name", "fresh0")]),
        )
        .await
        .map(|_| ())
    }

    async fn deploy(&self, system_id: &str, config: &ContainerConfig) -> Result<()> {
        // cloud-init: the consumer password (alphanumeric by their charset
        // rule — injection-proof) and optional key, applied by the region
        // at first boot instead of a seed ISO.
        let mut user_data = String::from("#cloud-config\nssh_pwauth: true\n");
        user_data.push_str(&format!(
            "chpasswd:\n  expire: false\n  users:\n    - name: root\n      password: {}\n      type: text\n",
            config.password
        ));
        if let Some(key) = &config.ssh_key {
            user_data.push_str(&format!("ssh_authorized_keys:\n  - {key}\n"));
        }
        self.send(
            self.http
                .post(self.machine_url(system_id))
                .query(&[("op", "deploy")])
                .form(&[("user_data", user_data)]),
        )
        .await
        .map(|_| ())
    }

    async fn delete(&self, system_id: &str) -> Result<()> {
        match self
            .send(self.http.delete(self.machine_url(system_id)))
            .await
        {
            Ok(_) => Ok(()),
            // Idempotent for the unconditional cleanup sweep.
            Err(e) if e.to_string().contains("404") => Ok(()),
            Err(e) => Err(e),
        }
    }

    async fn vm_host(&self, vm_host_id: u32) -> Result<MaasVmHost> {
        let url = self
            .base
            .join(&format!("vm-hosts/{vm_host_id}/"))
            .expect("valid base");
        self.send(self.http.get(url))
            .await?
            .json()
            .await
            .context("decode vm-host capacity")
    }
}

/// Render a "consumer:token:secret" MAAS api key as an OAuth1 header with
/// PLAINTEXT signatures (TLS-terminated region access).
fn oauth1_plaintext(api_key: &str) -> String {
    let parts: Vec<&str> = api_key.split(':').collect();
    let (ck, token, secret) = match parts.as_slice() {
        [ck, token, secret] => (*ck, *token, *secret),
        _ => ("", "", ""),
    };
    format!(
        "OAuth oauth_version=\"1.0\", oauth_signature_method=\"PLAINTEXT\", \
         oauth_consumer_key=\"{ck}\", oauth_token=\"{token}\", \
         oauth_signature=\"{secret}&\""
    )
}

/// A locally-administered unicast MAC (`02:…`), unique per call with
/// overwhelming probability — enough for the fresh-interface trick.
fn fresh_local_mac() -> String {
    let mut rng = rand::thread_rng();
    let bytes: [u8; 6] = core::array::from_fn(|_| rng.gen());
    format!(
        "02:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]
    )
}

/// MAAS-backed workload store: machines named `paygress-<id>` on one VM
/// host.
pub struct MaasBackend {
    client: MaasClient,
    vm_host_id: u32,
}

impl MaasBackend {
    pub fn new(client: MaasClient, vm_host_id: u32) -> Self {
        Self { client, vm_host_id }
    }

    /// Startup fail-fast, mirroring `KvmBackend::check_kvm_available`: a
    /// provider that cannot reach its region must refuse to start before
    /// any consumer commits a token.
    pub async fn check_region_reachable(client: &MaasClient) -> Result<()> {
        match client.machines().await {
            Ok(_) => Ok(()),
            Err(e) => bail!("MAAS region unreachable: {e}"),
        }
    }
}

#[async_trait]
impl ComputeBackend for MaasBackend {
    async fn find_available_id(&self, range_start: u32, range_end: u32) -> Result<u32> {
        let used: std::collections::HashSet<u32> = self
            .client
            .machines()
            .await?
            .iter()
            .filter_map(|m| crate::compute::id_from_container_name(&m.hostname))
            .collect();
        for id in range_start..=range_end {
            if !used.contains(&id) {
                return Ok(id);
            }
        }
        bail!("no available id in range {range_start}..={range_end}");
    }

    async fn create_container(&self, config: &ContainerConfig) -> Result<String> {
        let hostname = container_name(config.id);
        let machine = self
            .client
            .compose(
                self.vm_host_id,
                &hostname,
                config.cpu_cores,
                config.memory_mb,
                config.storage_gb,
            )
            .await?;
        let system_id = machine.system_id.clone();

        // Deploy-without-commissioning: fresh hardware identity, then the
        // region's "fixed" path lands the machine in Ready without a
        // commissioning run.
        self.client.op(&system_id, "mark_broken").await?;
        self.client.attach_fresh_interface(&system_id).await?;
        self.client.op(&system_id, "mark_fixed").await?;

        for _ in 0..READY_POLL_ATTEMPTS {
            if let Some(m) = self.client.machine(&system_id).await? {
                if m.is_ready() {
                    break;
                }
            }
            tokio::time::sleep(READY_POLL_INTERVAL).await;
        }

        self.client.deploy(&system_id, config).await?;
        if let Some(m) = self.client.machine(&system_id).await? {
            if !m.is_deploying_or_deployed() {
                bail!(
                    "deploy not accepted for {hostname}: state {}",
                    m.status_name
                );
            }
        }
        Ok(system_id)
    }

    async fn start_container(&self, id: u32) -> Result<()> {
        let Some(m) = self.machine_by_id(id).await? else {
            return Ok(());
        };
        self.client.op(&m.system_id, "power_on").await
    }

    async fn stop_container(&self, id: u32) -> Result<()> {
        let Some(m) = self.machine_by_id(id).await? else {
            return Ok(());
        };
        self.client.op(&m.system_id, "power_off").await
    }

    async fn delete_container(&self, id: u32) -> Result<()> {
        let Some(m) = self.machine_by_id(id).await? else {
            return Ok(());
        };
        self.client.delete(&m.system_id).await
    }

    async fn get_node_status(&self) -> Result<NodeStatus> {
        let host = self.client.vm_host(self.vm_host_id).await?;
        Ok(NodeStatus {
            cpu_usage: if host.cpu_total > 0.0 {
                (host.cpu_used / host.cpu_total * 100.0).min(100.0)
            } else {
                0.0
            },
            memory_used: host.memory_used,
            memory_total: host.memory_total,
            // VM-host pooled storage is not a node disk reading; zeroes
            // read as "unmeasured" by `NodeStatus::free_disk_bytes`, which
            // keeps the spawn-time headroom gate honest.
            disk_used: 0,
            disk_total: 0,
        })
    }

    async fn get_container_ip(&self, id: u32) -> Result<Option<String>> {
        self.machine_by_id(id)
            .await
            .map(|m| m.and_then(|m| m.ip_addresses.first().cloned()))
    }

    async fn get_container_status(&self, id: u32) -> Result<ContainerStatus> {
        match self.machine_by_id(id).await? {
            None => Ok(ContainerStatus::Absent),
            Some(m) if m.is_deploying_or_deployed() => Ok(ContainerStatus::Running),
            Some(_) => Ok(ContainerStatus::Stopped),
        }
    }
}

impl MaasBackend {
    /// Resolve `paygress-<id>` to its machine via the hostname (system ids
    /// are opaque; the name is the durable handle — same as every other
    /// backend).
    async fn machine_by_id(&self, id: u32) -> Result<Option<MaasMachine>> {
        let name = container_name(id);
        self.client
            .machines()
            .await
            .map(|ms| ms.into_iter().find(|m| m.hostname == name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_string_contains, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(id: u32) -> ContainerConfig {
        ContainerConfig {
            id,
            name: container_name(id),
            image: "ubuntu".into(),
            cpu_cores: 2,
            memory_mb: 2048,
            storage_gb: 20,
            password: "alphabetic12".into(),
            ssh_key: None,
            host_port: None,
            template_ports: vec![],
            template_env: Default::default(),
            extra_runtime_args: vec![],
            data_path: None,
            volume_encryption_key: None,
        }
    }

    fn machine_json(id: &str, hostname: &str, status: &str, ips: &[&str]) -> serde_json::Value {
        serde_json::json!({
            "system_id": id,
            "hostname": hostname,
            "status_name": status,
            "ip_addresses": ips,
        })
    }

    fn client_for(server: &MockServer) -> MaasClient {
        MaasClient::new(&format!("{}/MAAS/api/2.0", server.uri()), "ck:tok:sec").unwrap()
    }

    #[tokio::test]
    async fn create_container_runs_the_no_commission_sequence() {
        let server = MockServer::start().await;
        // wiremock breaks priority ties by insertion order (first mounted
        // wins): the one-shot "Ready" reader mounts FIRST and answers the
        // mark-fixed readiness poll; once exhausted, every later poll
        // falls through to the unlimited "Deploying" reader (the
        // post-deploy acceptance check), mounted last.
        let ready_reader = Mock::given(method("GET"))
            .and(path("/MAAS/api/2.0/machines/sys-1/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(machine_json(
                "sys-1",
                "paygress-7",
                "Ready",
                &[],
            )))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        let compose = Mock::given(method("POST"))
            .and(path("/MAAS/api/2.0/vm-hosts/1/compose/"))
            .and(body_string_contains("hostname=paygress-7"))
            .respond_with(ResponseTemplate::new(200).set_body_json(machine_json(
                "sys-1",
                "paygress-7",
                "New",
                &[],
            )))
            .expect(1)
            .mount(&server)
            .await;
        let mark_broken = Mock::given(method("POST"))
            .and(path("/MAAS/api/2.0/machines/sys-1/"))
            .and(query_param("op", "mark_broken"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .expect(1)
            .mount(&server)
            .await;
        let attach = Mock::given(method("POST"))
            .and(path("/MAAS/api/2.0/machines/sys-1/interfaces/"))
            .and(body_string_contains("mac_address=02%3A"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .expect(1)
            .mount(&server)
            .await;
        let mark_fixed = Mock::given(method("POST"))
            .and(path("/MAAS/api/2.0/machines/sys-1/"))
            .and(query_param("op", "mark_fixed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(machine_json(
                "sys-1",
                "paygress-7",
                "Ready",
                &[],
            )))
            .expect(1)
            .mount(&server)
            .await;
        let deploy = Mock::given(method("POST"))
            .and(path("/MAAS/api/2.0/machines/sys-1/"))
            .and(query_param("op", "deploy"))
            .and(body_string_contains("ssh_pwauth"))
            .and(body_string_contains("alphabetic12"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .expect(1)
            .mount(&server)
            .await;
        let _ = (
            ready_reader,
            compose,
            mark_broken,
            attach,
            mark_fixed,
            deploy,
        );
        Mock::given(method("GET"))
            .and(path("/MAAS/api/2.0/machines/sys-1/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(machine_json(
                "sys-1",
                "paygress-7",
                "Deploying",
                &[],
            )))
            .mount(&server)
            .await;

        let backend = MaasBackend::new(client_for(&server), 1);
        let vmid = backend.create_container(&config(7)).await.unwrap();

        assert_eq!(vmid, "sys-1");
        // Every op in the deploy-without-commissioning sequence fired.
        server.verify().await;
    }

    #[tokio::test]
    async fn find_available_id_skips_taken_paygress_names() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/MAAS/api/2.0/machines/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                machine_json("sys-1", "paygress-1", "Deployed", &["10.50.0.11"]),
                machine_json("sys-2", "paygress-3", "Ready", &[]),
                machine_json("sys-3", "other-host", "Deployed", &[]),
            ])))
            .mount(&server)
            .await;

        let backend = MaasBackend::new(client_for(&server), 1);
        assert_eq!(backend.find_available_id(1, 10).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn delete_is_idempotent_for_reaped_machines() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/MAAS/api/2.0/machines/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&server)
            .await;

        let backend = MaasBackend::new(client_for(&server), 1);
        backend.delete_container(9).await.unwrap(); // no such name → Ok
    }

    #[tokio::test]
    async fn status_and_ip_come_from_the_machine_record() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/MAAS/api/2.0/machines/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                machine_json("sys-1", "paygress-4", "Deployed", &["10.50.0.44"]),
                machine_json("sys-2", "paygress-5", "New", &[]),
            ])))
            .mount(&server)
            .await;

        let backend = MaasBackend::new(client_for(&server), 1);
        assert_eq!(
            backend.get_container_ip(4).await.unwrap().as_deref(),
            Some("10.50.0.44")
        );
        assert_eq!(backend.get_container_ip(5).await.unwrap(), None);
        assert_eq!(
            backend.get_container_status(4).await.unwrap(),
            ContainerStatus::Running
        );
        assert_eq!(
            backend.get_container_status(5).await.unwrap(),
            ContainerStatus::Stopped
        );
        assert_eq!(
            backend.get_container_status(6).await.unwrap(),
            ContainerStatus::Absent
        );
    }

    #[tokio::test]
    async fn node_status_reads_vm_host_capacity_and_reports_unmeasured_disk() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/MAAS/api/2.0/vm-hosts/1/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "cpu_total": 8.0,
                "cpu_used": 2.0,
                "memory_total": 16384,
                "memory_used": 4096,
            })))
            .mount(&server)
            .await;

        let backend = MaasBackend::new(client_for(&server), 1);
        let status = backend.get_node_status().await.unwrap();

        assert!((status.cpu_usage - 25.0).abs() < 0.01);
        assert_eq!(status.memory_total, 16384);
        // Pooled storage is not a node disk reading: zeroes, not a lie.
        assert_eq!(status.free_disk_bytes(), None);
        assert!(status.has_disk_headroom(10));
    }

    #[tokio::test]
    async fn startup_check_fails_fast_when_the_region_is_unreachable() {
        // Port 1 refuses connections: nothing listens there.
        let client = MaasClient::new("http://127.0.0.1:1/MAAS/api/2.0", "ck:tok:sec").unwrap();
        assert!(MaasBackend::check_region_reachable(&client).await.is_err());
    }

    #[test]
    fn oauth_header_shape() {
        let h = oauth1_plaintext("ck:tok:sec");
        assert!(h.contains("oauth_signature_method=\"PLAINTEXT\""));
        assert!(h.ends_with("oauth_signature=\"sec&\""));
    }

    #[test]
    fn fresh_macs_are_locally_administered_and_unique() {
        let a = fresh_local_mac();
        let b = fresh_local_mac();
        assert!(a.starts_with("02:"));
        assert_ne!(a, b);
    }
}
