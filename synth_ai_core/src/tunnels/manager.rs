use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use tokio::task::JoinHandle;

use crate::tunnels::cloudflared::resolve_hostname_with_explicit_resolvers;
use crate::tunnels::errors::TunnelError;
use crate::tunnels::gateway::get_gateway;
use crate::tunnels::lease_client::LeaseClient;
use crate::tunnels::types::{LeaseInfo, LeaseState, TunnelBackend, TunnelHandle};
use crate::tunnels::{cloudflared, connector::get_connector};

const DEFAULT_LOCAL_READY_TIMEOUT: f64 = 30.0;
const DEFAULT_PUBLIC_READY_TIMEOUT: f64 = 60.0;

fn client_instance_path() -> Result<PathBuf, TunnelError> {
    let home = std::env::var("HOME").map_err(|_| TunnelError::config("HOME not set"))?;
    Ok(PathBuf::from(home)
        .join(".synth")
        .join("client_instance_id"))
}

fn get_client_instance_id() -> String {
    if let Ok(path) = client_instance_path() {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            let trimmed = contents.trim().to_string();
            if trimmed.len() >= 8 {
                return trimmed;
            }
        }
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let id = format!("client-{}", uuid::Uuid::new_v4().simple());
        let _ = std::fs::write(&path, &id);
        return id;
    }
    format!("session-{}", uuid::Uuid::new_v4().simple())
}

pub struct TunnelManager {
    api_key: Option<String>,
    backend_url: Option<String>,
    client_instance_id: String,
    client: Option<LeaseClient>,
    active_handles: HashMap<String, TunnelHandle>,
    heartbeat_tasks: HashMap<String, JoinHandle<()>>,
}

impl TunnelManager {
    pub fn new(api_key: Option<String>, backend_url: Option<String>) -> Self {
        Self {
            api_key,
            backend_url,
            client_instance_id: get_client_instance_id(),
            client: None,
            active_handles: HashMap::new(),
            heartbeat_tasks: HashMap::new(),
        }
    }

    async fn client(&mut self) -> Result<&LeaseClient, TunnelError> {
        if self.client.is_none() {
            let key = self
                .api_key
                .clone()
                .or_else(|| std::env::var("SYNTH_API_KEY").ok())
                .ok_or_else(|| TunnelError::config("SYNTH_API_KEY missing"))?;
            let backend = self
                .backend_url
                .clone()
                .unwrap_or_else(|| "https://api.usesynth.ai".to_string());
            self.client = Some(LeaseClient::new(key, backend, 30)?);
        }
        Ok(self.client.as_ref().unwrap())
    }

    pub async fn open(
        &mut self,
        local_port: u16,
        local_host: &str,
        verify_local: bool,
        verify_public: bool,
        _progress: bool,
        public_api_key: Option<String>,
    ) -> Result<TunnelHandle, TunnelError> {
        // Capture values before borrowing self mutably
        let client_instance_id = self.client_instance_id.clone();
        let api_key = self.api_key.clone();
        let client = self.client().await?;
        let mut lease = client
            .create_lease(
                &client_instance_id,
                local_host,
                local_port,
                None,
                3600,
                true,
                None,
            )
            .await?;

        {
            let gateway = get_gateway();
            let mut guard = gateway.lock();
            if !guard.is_running() {
                guard.start(true).await?;
            }
            guard.add_route(&lease.route_prefix, local_host, local_port);
        }

        if verify_local {
            cloudflared::wait_for_health_check(
                local_host,
                local_port,
                public_api_key.clone(),
                DEFAULT_LOCAL_READY_TIMEOUT,
            )
            .await?;
        }

        // Ensure connector
        {
            let connector = get_connector();
            let mut guard = connector.lock();
            guard
                .start(&lease.tunnel_token, Duration::from_secs(60), false)
                .await?;
            guard.register_lease(&lease.lease_id);
        }

        if verify_public {
            verify_public_ready(&lease, public_api_key.or(api_key)).await?;
        }

        // Heartbeat once
        let (action, _) = client
            .heartbeat(&lease.lease_id, true, true, true, None)
            .await?;
        if action == "restart_connector" {
            // noop for now
        }
        lease.state = LeaseState::Active;

        let connector_status = get_connector().lock().status();
        let gateway_status = get_gateway().lock().status();
        let handle = TunnelHandle {
            url: lease.public_url.clone(),
            hostname: lease.hostname.clone(),
            local_port,
            lease: Some(lease.clone()),
            connector: Some(connector_status),
            gateway: Some(gateway_status),
            backend: TunnelBackend::CloudflareManagedLease,
            process_id: None,
        };
        self.active_handles
            .insert(lease.lease_id.clone(), handle.clone());
        let lease_id = lease.lease_id.clone();
        let client = self.client().await?.clone();
        let connector = get_connector();
        let gateway = get_gateway();
        let task = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let connected = connector.lock().is_connected();
                let gateway_ready = gateway.lock().is_running();
                let _ = client
                    .heartbeat(&lease_id, connected, gateway_ready, true, None)
                    .await;
            }
        });
        self.heartbeat_tasks.insert(lease.lease_id.clone(), task);
        Ok(handle)
    }

    pub async fn close(&mut self, lease_id: &str) -> Result<(), TunnelError> {
        if let Some(task) = self.heartbeat_tasks.remove(lease_id) {
            task.abort();
        }
        if let Some(handle) = self.active_handles.remove(lease_id) {
            if let Some(lease) = handle.lease {
                get_gateway().lock().remove_route(&lease.route_prefix);
            }
        }
        let client = self.client().await?;
        let _ = client.release(lease_id).await;
        {
            let connector = get_connector();
            connector.lock().unregister_lease(lease_id);
        }
        Ok(())
    }
}

async fn verify_public_ready(
    lease: &LeaseInfo,
    api_key: Option<String>,
) -> Result<(), TunnelError> {
    let hostname = &lease.hostname;
    let route = &lease.route_prefix;
    let ready_url = format!("https://{hostname}{route}/__synth/ready");
    let deadline =
        std::time::Instant::now() + Duration::from_secs_f64(DEFAULT_PUBLIC_READY_TIMEOUT);
    let mut resolved_ip = resolve_hostname_with_explicit_resolvers(hostname)
        .await
        .ok();
    while std::time::Instant::now() < deadline {
        let mut builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .danger_accept_invalid_certs(true);
        if let Some(ip) = resolved_ip {
            builder = builder.resolve(hostname, (ip, 443).into());
        }
        let client = builder
            .build()
            .map_err(|e| TunnelError::dns(e.to_string()))?;
        let mut req = client.get(&ready_url);
        if let Some(key) = api_key.clone() {
            req = req.header("X-API-Key", key);
        }
        match req.send().await {
            Ok(resp) => {
                if resp.status().as_u16() == 200 {
                    return Ok(());
                }
            }
            Err(_) => {
                resolved_ip = None;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    Ok(())
}

static MANAGER: Lazy<Mutex<TunnelManager>> =
    Lazy::new(|| Mutex::new(TunnelManager::new(None, None)));

pub fn get_manager(
    api_key: Option<String>,
    backend_url: Option<String>,
) -> &'static Mutex<TunnelManager> {
    if api_key.is_some() || backend_url.is_some() {
        let mut guard = MANAGER.lock();
        guard.api_key = api_key.or_else(|| guard.api_key.clone());
        guard.backend_url = backend_url.or_else(|| guard.backend_url.clone());
    }
    &MANAGER
}
