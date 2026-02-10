pub mod cloudflared;
pub mod connector;
pub mod errors;
pub mod gateway;
pub mod lease_client;
pub mod manager;
pub mod ports;
pub mod synth_tunnel;
pub mod types;

use crate::tunnels::errors::TunnelError;
use crate::tunnels::types::{TunnelBackend, TunnelHandle};

pub async fn open_tunnel(
    backend: TunnelBackend,
    local_port: u16,
    api_key: Option<String>,
    backend_url: Option<String>,
    local_api_key: Option<String>,
    verify_local: bool,
    verify_dns: bool,
    progress: bool,
) -> Result<TunnelHandle, TunnelError> {
    match backend {
        TunnelBackend::Localhost => {
            let url = format!("http://localhost:{local_port}");
            Ok(TunnelHandle {
                url: url.clone(),
                hostname: "localhost".to_string(),
                local_port,
                lease: None,
                connector: None,
                gateway: None,
                backend,
                process_id: None,
            })
        }
        TunnelBackend::CloudflareManagedLease => {
            let manager = manager::get_manager(api_key.clone(), backend_url.clone());
            let mut guard = manager.lock();
            guard
                .open(
                    local_port,
                    "127.0.0.1",
                    verify_local,
                    verify_dns,
                    progress,
                    local_api_key,
                )
                .await
        }
        TunnelBackend::CloudflareManaged => {
            let key = api_key.ok_or_else(|| TunnelError::config("api_key is required"))?;
            let data = cloudflared::rotate_tunnel(&key, local_port, backend_url.clone()).await?;
            let hostname = data.hostname.clone();
            let token = data.tunnel_token.clone();
            let url = format!("https://{hostname}");
            let proc = cloudflared::open_managed_tunnel_with_connection_wait(&token, 30.0).await?;
            let process_id = cloudflared::track_process(proc);
            if verify_dns {
                cloudflared::verify_tunnel_dns_resolution(
                    &url,
                    "tunnel",
                    60.0,
                    local_api_key.clone(),
                )
                .await?;
            }
            Ok(TunnelHandle {
                url,
                hostname,
                local_port,
                lease: None,
                connector: None,
                gateway: None,
                backend,
                process_id: Some(process_id),
            })
        }
        TunnelBackend::SynthTunnel => {
            return Err(TunnelError::config(
                "SynthTunnel uses synth_tunnel::start_agent() directly; do not call open_tunnel()",
            ));
        }
        TunnelBackend::CloudflareQuick => {
            let (url, proc) = cloudflared::open_quick_tunnel_with_dns_verification(
                local_port,
                10.0,
                verify_dns,
                local_api_key.clone(),
            )
            .await?;
            let process_id = cloudflared::track_process(proc);
            let hostname = url
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .trim_end_matches('/')
                .to_string();
            Ok(TunnelHandle {
                url,
                hostname,
                local_port,
                lease: None,
                connector: None,
                gateway: None,
                backend,
                process_id: Some(process_id),
            })
        }
    }
}
