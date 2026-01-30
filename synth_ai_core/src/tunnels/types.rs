use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseState {
    Pending,
    Active,
    Released,
    Expired,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectorState {
    Stopped,
    Starting,
    Connected,
    Disconnected,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatewayState {
    Stopped,
    Starting,
    Running,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaseInfo {
    pub lease_id: String,
    pub managed_tunnel_id: String,
    pub hostname: String,
    pub route_prefix: String,
    pub public_url: String,
    pub local_host: String,
    pub local_port: u16,
    pub expires_at: DateTime<Utc>,
    pub tunnel_token: String,
    pub access_client_id: Option<String>,
    pub access_client_secret: Option<String>,
    pub gateway_port: u16,
    pub state: LeaseState,
    pub diagnostics_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorStatus {
    pub state: ConnectorState,
    pub pid: Option<u32>,
    pub connected_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayStatus {
    pub state: GatewayState,
    pub port: u16,
    pub routes: Vec<(String, String, u16)>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelHandle {
    pub url: String,
    pub hostname: String,
    pub local_port: u16,
    pub lease: Option<LeaseInfo>,
    pub connector: Option<ConnectorStatus>,
    pub gateway: Option<GatewayStatus>,
    pub backend: TunnelBackend,
    pub process_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    pub lease_id: String,
    pub tunnel_id: String,
    pub client_instance_id: String,
    pub hostname: String,
    pub connector_state: String,
    pub gateway_state: String,
    pub lease_state: String,
    pub last_error: Option<String>,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TunnelBackend {
    Localhost,
    CloudflareQuick,
    CloudflareManaged,
    CloudflareManagedLease,
    SynthTunnel,
}

#[derive(Debug, Clone)]
pub struct SynthTunnelConfig {
    pub ws_url: String,
    pub agent_token: String,
    pub lease_id: String,
    pub local_host: String,
    pub local_port: u16,
    pub public_url: String,
    pub worker_token: String,
    pub local_api_keys: Vec<String>,
    pub max_inflight: usize,
}
