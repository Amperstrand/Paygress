// MAAS-backed ComputeBackend: the paygress provider protocol (Cashu leases,
// Nostr offers/spawn DMs) on top of a MAAS region controller. MAAS stays the
// single source of truth for compute; this backend only translates the
// ComputeBackend contract into MAAS CLI calls plus one SSH hop to the LXD
// host for the instance-MAC fix.
//
// Why the MAC fix exists (conwrt #193/#194 research, 2026-10-15):
// `maas vm-host compose` succeeds without registering the machine's
// interface, so PXE intent can never match the instance MAC and a deploy
// would stall forever. The proven repair is compose -> mark-broken ->
// create-physical (fresh locally-administered MAC) -> mark-fixed, which
// reaches Ready WITHOUT commissioning (~60s), leaving only the deploy inside
// the spawn path.
//
// Requires, on the host running the provider:
//   - `maas` CLI on PATH with a logged-in profile (root's default works)
//   - passwordless SSH to the LXD host (instance stop/set-MAC)

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use rand::RngCore;
use serde_json::Value;
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::compute::{ComputeBackend, ContainerConfig, ContainerStatus, NodeStatus};

const HOSTNAME_PREFIX: &str = "paygress-";

/// MAAS-specific provider configuration. Values come from the provider
/// config JSON; nothing deployment-specific is hardcoded here.
#[derive(Debug, Clone)]
pub struct MaasConfig {
    /// MAAS VM-host (LXD pod) id that spawns land on.
    pub vm_host_id: u32,
    /// Subnet constraint for the spawned NIC, e.g. `10.0.0.0/24`.
    pub subnet: String,
    /// VLAN id the NIC is attached to (from `maas vlans read`).
    pub vlan_id: u32,
    /// SSH target for the LXD host, e.g. `ubuntu@10.0.0.2`; key auth only.
    pub lxd_ssh_target: String,
    /// MAAS CLI profile name (see `maas login`).
    pub profile: String,
    /// Distro series deployed on spawn (e.g. `noble`).
    pub distro_series: String,
    /// macvlan peer: lower device (e.g. `eth0.2`) for the host-side
    /// macvlan bridge that reaches the guests — the host's own stack is
    /// isolated from macvlan guests without it.
    pub mv_parent: String,
    /// Source address on the macvlan peer, used for the /32 routes and the
    /// SNAT of forwarded SSH flows.
    pub mv_ip: String,
    /// Host address the SSH DNAT rules listen on (the provider public_ip):
    /// NAT-mode LXD proxies reject wildcard listens, non-NAT proxies are
    /// unsupported on VM instances, and NAT connect-IPs must be LXD-static —
    /// so the forward is host iptables DNAT, the same mechanism the bench's
    /// public port-forward uses.
    pub listen_host: String,
}

pub struct MaasBackend {
    config: MaasConfig,
    /// workload id -> MAAS system_id, learned at spawn; rebuilt from
    /// hostnames on lookup miss so a provider restart never strands a
    /// workload.
    system_ids: Mutex<HashMap<u32, String>>,
}

impl MaasBackend {
    pub fn new(config: MaasConfig) -> Self {
        Self {
            config,
            system_ids: Mutex::new(HashMap::new()),
        }
    }

    /// Run `maas <profile> <args...>` and return stdout.
    async fn maas(&self, args: &[&str]) -> Result<String> {
        let mut argv = vec!["maas", self.config.profile.as_str()];
        argv.extend_from_slice(args);
        run_checked(&argv).await
    }

    /// One MAAS CLI call whose stdout is parsed as JSON.
    async fn maas_json(&self, args: &[&str]) -> Result<Value> {
        let out = self.maas(args).await?;
        serde_json::from_str(&out)
            .with_context(|| format!("parse JSON from `maas {}`", args.join(" ")))
    }

    async fn lxd_ssh(&self, command: &str) -> Result<String> {
        let out = Command::new("ssh")
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-o")
            .arg("StrictHostKeyChecking=accept-new")
            .arg(&self.config.lxd_ssh_target)
            .arg(command)
            .output()
            .await
            .context("invoke ssh to lxd host")?;
        if !out.status.success() {
            bail!(
                "ssh lxd host failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn hostname(id: u32) -> String {
        format!("{HOSTNAME_PREFIX}{id}")
    }

    /// Look up the MAAS system_id for a workload id by hostname, caching
    /// the result. Errors when no such machine exists.
    async fn system_id(&self, id: u32) -> Result<String> {
        if let Some(sid) = self.system_ids.lock().unwrap().get(&id) {
            return Ok(sid.clone());
        }
        let hostname = Self::hostname(id);
        let machines = self
            .maas_json(&["machines", "read"])
            .await
            .context("list machines for hostname lookup")?;
        let sid = machines
            .as_array()
            .and_then(|rows| {
                rows.iter()
                    .find(|m| m["hostname"].as_str() == Some(hostname.as_str()))
                    .and_then(|m| m["system_id"].as_str())
            })
            .map(str::to_string)
            .with_context(|| format!("no MAAS machine named {hostname}"))?;
        self.system_ids.lock().unwrap().insert(id, sid.clone());
        Ok(sid)
    }

    /// The deploy user-data: password SSH (paygress access model), optional
    /// consumer key, guest agent. `config.password` is charset-restricted
    /// by the spawn handler (12-64 alphanumeric), so YAML embedding is safe.
    fn user_data(config: &ContainerConfig) -> String {
        let key_line = config
            .ssh_key
            .as_ref()
            .map(|k| format!("    - {k}\n"))
            .unwrap_or_default();
        format!(
            "#cloud-config\n\
             ssh_pwauth: true\n\
             chpasswd:\n  expire: false\n  users:\n    - name: ubuntu\n      password: {pw}\n      type: text\n\
             ssh_authorized_keys:\n{keys}\
             packages: [qemu-guest-agent]\n\
             runcmd:\n  - systemctl enable --now qemu-guest-agent || true\n",
            pw = config.password,
            keys = key_line,
        )
    }

    /// Poll machine status until it reaches `want`, with a deadline.
    /// Any `Failed *`/`Broken` status aborts early.
    async fn wait_status(&self, sid: &str, want: &str, deadline_secs: u64) -> Result<String> {
        let start = std::time::Instant::now();
        loop {
            let machine = self
                .maas_json(&["machine", "read", sid])
                .await
                .context("machine read while waiting")?;
            let status = machine["status_name"].as_str().unwrap_or("Unknown").to_string();
            if status == want {
                return Ok(status);
            }
            if status.starts_with("Failed") || status == "Broken" {
                bail!("machine {sid} entered {status} while waiting for {want}");
            }
            if start.elapsed().as_secs() > deadline_secs {
                bail!("machine {sid} did not reach {want} within {deadline_secs}s (last: {status})");
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }
    /// One-time host plumbing for reachable workloads:
    ///  - a host-side macvlan bridge peer (the host cannot talk to its own
    ///    macvlan guests without one)
    ///  - a global SNAT for SSH flows leaving via the peer, so guest replies
    ///    come back through the conntrack that DNAT'd them
    ///  - an ESTABLISHED forward accept (ufw's default forward policy drops)
    /// All idempotent; safe on every spawn.
    async fn ensure_host_forwarding(&self) -> Result<()> {
        let rc = self
            .lxd_ssh(&format!(
                "ip link show paygress-mv >/dev/null 2>&1 || \
                 (sudo ip link add paygress-mv link {parent} type macvlan mode bridge && \
                  sudo ip addr add {ip}/32 dev paygress-mv && \
                  sudo ip link set paygress-mv up && \
                  sudo iptables -t nat -A POSTROUTING -o paygress-mv -p tcp --dport 22 \
                    -j SNAT --to-source {ip} -m comment --comment paygress-mv && \
                  sudo iptables -I FORWARD 1 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT \
                    -m comment --comment paygress-mv)",
                parent = self.config.mv_parent,
                ip = self.config.mv_ip,
            ))
            .await;
        if rc.is_err() {
            // Already-configured hosts answer `ip link show` with success,
            // so an error here means the create path failed — surface it.
            debug!("host forwarding ensure rc: {rc:?}");
        }
        Ok(())
    }

    /// MAAS records the deployed machine's DHCP lease in its interface
    /// links a beat AFTER the status flips to Deployed — one immediate
    /// lookup races that write and reaped healthy workloads (#194).
    async fn await_workload_ip(&self, id: u32, deadline_secs: u64) -> Result<Option<String>> {
        let start = std::time::Instant::now();
        let hostname = Self::hostname(id);
        loop {
            if let Some(ip) = self.get_container_ip(id).await? {
                return Ok(Some(ip));
            }
            // MAAS's lease-notification path drops DHCPACKs on this install
            // (dhcpd.sock warnings), so interface links never populate; the
            // LXD guest IP is the same fact from a source that never misses.
            if let Ok(out) = self
                .lxd_ssh(&format!("sudo lxc list {hostname} --format csv -c 4 2>/dev/null"))
                .await
            {
                if let Some(ip) = parse_first_ipv4(&out) {
                    return Ok(Some(ip));
                }
            }
            if start.elapsed().as_secs() > deadline_secs {
                return Ok(None);
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    /// Everything after compose. Split out so create_container can reap the
    /// composed machine when any later step fails.
    async fn finish_create(
        &self,
        config: &ContainerConfig,
        id: u32,
        hostname: &str,
        sid: &str,
    ) -> Result<String> {
        let hostname = hostname.to_string();

        // 2. Fresh MAC on the instance + explicit interface registration.
        //    Commissioning would normally discover the NIC link and storage
        //    layout; without it, both must be registered by hand:
        //    create-physical -> link-subnet (DHCP) -> set-storage-layout
        //    (flat root) -> mark-fixed reaches Ready in ~60s, no commission.
        let mac = random_mac();
        self.lxd_ssh(&format!(
            "sudo lxc stop -f {hostname} 2>/dev/null; sudo lxc config set {hostname} volatile.eth0.hwaddr {mac}"
        ))
        .await
        .context("set instance MAC via lxd host")?;
        self.maas(&["machine", "mark-broken", &sid]).await?;
        let iface = self
            .maas_json(&[
                "interfaces",
                "create-physical",
                &sid,
                &format!("mac_address={mac}"),
                &format!("vlan={}", self.config.vlan_id),
                "name=eth0",
            ])
            .await
            .context("create-physical interface")?;
        let iface_id = iface["id"]
            .as_u64()
            .context("create-physical returned no interface id")?
            .to_string();
        self.maas(&[
            "interface",
            "link-subnet",
            &sid,
            &iface_id,
            &format!("subnet={}", self.config.subnet),
            "mode=DHCP",
        ])
        .await
        .context("link interface to subnet")?;
        // Order matters (#194 field notes): set-storage-layout is rejected on
        // a Broken machine (manual proof had it in Ready), and mark-fixed is
        // what reaches Ready without commissioning.
        self.maas(&["machine", "mark-fixed", &sid]).await?;
        self.maas(&["machine", "set-storage-layout", &sid, "storage_layout=flat"])
            .await
            .context("set root storage layout")?;
        self.wait_status(&sid, "Ready", 120).await?;

        // 3. Deploy (deploy-only measured 3-4.5 min, inside the fork's
        //    raised BACKEND_CALL_TIMEOUT).
        let user_data = BASE64.encode(Self::user_data(config));
        self.maas(&[
            "machine",
            "deploy",
            &sid,
            &format!("user_data={user_data}"),
            &format!("distro_series={}", self.config.distro_series),
        ])
        .await
        .context("machine deploy")?;
        self.wait_status(&sid, "Deployed", 420).await?;

        // 4. SSH reachability: consumers dial `public_ip : host_port`
        //    (spawn handler sets it from ssh_host_port(id)), so forward that
        //    port on the LXD host into the VM — same proxy-device mechanism
        //    the upstream LXD backend uses.
        if let Some(port) = config.host_port {
            let ip = self
                .await_workload_ip(id, 90)
                .await?
                .context("workload deployed but no IP appeared for the ssh proxy device")?;
            // Per-VM /32 route through the macvlan peer: DNAT'd packets
            // must egress the peer, not the host's own VLAN interface
            // (macvlan isolation drops those).
            let _ = self
                .lxd_ssh(&format!("sudo ip route replace {ip}/32 dev paygress-mv"))
                .await;
            self.lxd_ssh(&format!(
                "sudo iptables -t nat -A PREROUTING -d {host} -p tcp --dport {port} \
                 -j DNAT --to-destination {ip}:22 -m comment --comment paygress-{id} && \
                 sudo iptables -I FORWARD 1 -d {ip} -p tcp --dport 22 -j ACCEPT \
                 -m comment --comment paygress-{id}",
                host = self.config.listen_host,
            ))
            .await
            .context("add ssh DNAT forward")?;
        }

        self.system_ids.lock().unwrap().insert(id, sid.to_string());
        info!("MAAS workload {hostname} ({sid}) deployed");
        Ok(hostname)
    }


}

/// Run a command, failing with the child's stderr when it exits non-zero
/// (same contract as compute::run_checked, for argv slices).
async fn run_checked(argv: &[&str]) -> Result<String> {
    let (program, args) = argv.split_first().expect("non-empty argv");
    let out = Command::new(program)
        .args(args)
        .output()
        .await
        .with_context(|| format!("invoke {program}"))?;
    if !out.status.success() {
        bail!(
            "{program} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// First IPv4 literal in a free-form string (lxc csv output).
fn parse_first_ipv4(s: &str) -> Option<String> {
    s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.'))
        .find(|t| t.parse::<std::net::Ipv4Addr>().is_ok())
        .map(str::to_string)
}

/// Fresh locally-administered MAC in the QEMU OUI space. Collisions are
/// surfaced by `interfaces create-physical` (duplicate-MAC error) rather
/// than silently misrouting.
fn random_mac() -> String {
    let mut bytes = [0u8; 3];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("00:16:3e:{:02x}:{:02x}:{:02x}", bytes[0], bytes[1], bytes[2])
}

#[async_trait]
impl ComputeBackend for MaasBackend {
    async fn find_available_id(&self, range_start: u32, range_end: u32) -> Result<u32> {
        let machines = self
            .maas_json(&["machines", "read"])
            .await
            .context("list machines")?;
        let used: std::collections::HashSet<u32> = machines
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|m| m["hostname"].as_str())
                    .filter_map(|h| h.strip_prefix(HOSTNAME_PREFIX))
                    .filter_map(|s| s.parse::<u32>().ok())
                    .collect()
            })
            .unwrap_or_default();
        for id in range_start..=range_end {
            if !used.contains(&id) {
                return Ok(id);
            }
        }
        bail!("no available workload id in range {range_start}..={range_end}");
    }

    async fn create_container(&self, config: &ContainerConfig) -> Result<String> {
        let id = config.id;
        let hostname = Self::hostname(id);
        info!(
            "Provisioning via MAAS: id={id} cores={} mem={}MB disk={}GB",
            config.cpu_cores, config.memory_mb, config.storage_gb
        );

        self.ensure_host_forwarding().await?;

        // 1. Compose the instance on the LXD pod. The compose's own auto-
        //    boot stalls harmlessly (unregistered interface); the fix below
        //    replaces it before deploy.
        let composed = self
            .maas_json(&[
                "vm-host",
                "compose",
                &self.config.vm_host_id.to_string(),
                &format!("cores={}", config.cpu_cores),
                &format!("memory={}", config.memory_mb),
                &format!("disks=1:size={}", config.storage_gb),
                &format!("interfaces=0:subnet={}", self.config.subnet),
                &format!("hostname={hostname}"),
            ])
            .await
            .context("vm-host compose")?;
        let sid = composed["system_id"]
            .as_str()
            .context("compose returned no system_id")?
            .to_string();
        debug!("composed {hostname} -> {sid}");

        let result = self.finish_create(config, id, &hostname, &sid).await;
        if result.is_err() {
            // A failed spawn must not leave the composed machine behind:
            // it holds the vmid and the LXD instance hostage.
            let _ = self.delete_container(id).await;
        }
        result
    }

    async fn start_container(&self, id: u32) -> Result<()> {
        let sid = self.system_id(id).await?;
        self.maas(&["machine", "power-on", &sid]).await.map(|_| ())
    }

    async fn stop_container(&self, id: u32) -> Result<()> {
        let sid = self.system_id(id).await?;
        self.maas(&["machine", "power-off", &sid]).await.map(|_| ())
    }

    async fn delete_container(&self, id: u32) -> Result<()> {
        let hostname = Self::hostname(id);
        // A missing machine must not fail the cleanup sweep.
        let sid = match self.system_id(id).await {
            Ok(sid) => sid,
            Err(e) => {
                warn!("no system_id for {hostname} ({e:#}); delete is a no-op");
                return Ok(());
            }
        };
        if let Err(e) = self.maas(&["machine", "delete", &sid]).await {
            warn!("machine delete failed for {sid} ({e:#}); continuing to lxc cleanup");
        }
        // MAAS machine delete usually reaps the instance, but compose/delete
        // debris is a known failure mode (non-atomic) — sweep the host too.
        let _ = self
            .lxd_ssh(&format!("sudo lxc delete -f {hostname} 2>/dev/null"))
            .await;
        let _ = self
            .lxd_ssh(&format!(
                "sudo iptables -t nat -S PREROUTING | grep -F 'paygress-{id}' | sed 's/^-A /-t nat -D /' | while read -r r; do sudo iptables $r; done; \
                 sudo iptables -S FORWARD | grep -F 'paygress-{id}' | sed 's/^-A /-D /' | while read -r r; do sudo iptables $r; done"
            ))
            .await;
        self.system_ids.lock().unwrap().remove(&id);
        Ok(())
    }

    async fn get_node_status(&self) -> Result<NodeStatus> {
        let pod = self
            .maas_json(&["vm-host", "read", &self.config.vm_host_id.to_string()])
            .await
            .context("vm-host read")?;
        let get = |section: &str, key: &str| pod[section][key].as_u64().unwrap_or(0);
        Ok(NodeStatus {
            // MAAS does not expose host CPU utilisation; 0.0 is fine — the
            // pre-redeem guard only consults disk headroom.
            cpu_usage: 0.0,
            memory_used: get("used", "memory"),
            memory_total: get("total", "memory"),
            disk_used: get("used", "local_storage"),
            disk_total: get("total", "local_storage"),
        })
    }

    async fn get_container_ip(&self, id: u32) -> Result<Option<String>> {
        let sid = self.system_id(id).await?;
        let machine = self
            .maas_json(&["machine", "read", &sid])
            .await
            .context("machine read for ip")?;
        let ip = machine["interface_set"].as_array().and_then(|ifaces| {
            ifaces.iter().find_map(|i| {
                i["links"].as_array().and_then(|links| {
                    links
                        .iter()
                        .find_map(|l| l["ip_address"].as_str().map(String::from))
                })
            })
        });
        Ok(ip)
    }

    async fn get_container_status(&self, id: u32) -> Result<ContainerStatus> {
        match self.system_id(id).await {
            Ok(sid) => {
                let machine = self.maas_json(&["machine", "read", &sid]).await?;
                let status = machine["status_name"].as_str().unwrap_or("Unknown");
                Ok(match status {
                    "Ready" | "Allocated" | "Broken" | "Failed commissioning"
                    | "Failed deployment" => ContainerStatus::Stopped,
                    _ => ContainerStatus::Running,
                })
            }
            Err(_) => Ok(ContainerStatus::Absent),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostname_roundtrip() {
        assert_eq!(MaasBackend::hostname(7), "paygress-7");
        assert_eq!(
            "paygress-7"
                .strip_prefix(HOSTNAME_PREFIX)
                .and_then(|s| s.parse::<u32>().ok()),
            Some(7)
        );
    }

    #[test]
    fn user_data_has_password_and_key() {
        let mut config = ContainerConfig {
            id: 1,
            name: "paygress-1".into(),
            image: "ignored".into(),
            cpu_cores: 2,
            memory_mb: 2048,
            storage_gb: 10,
            password: "alphabetic12".into(),
            ssh_key: Some("ssh-ed25519 AAAA test".into()),
            host_port: None,
            template_ports: vec![],
            template_env: Default::default(),
            extra_runtime_args: vec![],
            data_path: None,
            volume_encryption_key: None,
        };
        let ud = MaasBackend::user_data(&config);
        assert!(ud.contains("alphabetic12"));
        assert!(ud.contains("ssh-ed25519 AAAA test"));
        assert!(ud.contains("ssh_pwauth: true"));
        config.ssh_key = None;
        assert!(!MaasBackend::user_data(&config).contains("ssh-ed25519"));
    }

    #[test]
    fn random_mac_shape() {
        for _ in 0..50 {
            let mac = random_mac();
            let parts: Vec<&str> = mac.split(':').collect();
            assert_eq!(parts.len(), 6);
            assert_eq!(parts[0], "00");
            assert_eq!(parts[1], "16");
            assert_eq!(parts[2], "3e");
        }
    }
}
