use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::models::JsonMap;
use crate::openapi_paths;
use crate::transport::Transport;
use crate::types::Result;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SynthTunnelLeaseCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SynthTunnelLease {
    #[serde(default)]
    pub lease_id: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub worker_token: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone)]
pub struct TunnelsClient {
    transport: Arc<Transport>,
}

impl TunnelsClient {
    pub(crate) fn new(transport: Arc<Transport>) -> Self {
        Self { transport }
    }

    pub async fn create_lease(
        &self,
        request: &SynthTunnelLeaseCreateRequest,
    ) -> Result<SynthTunnelLease> {
        self.transport
            .post_json(openapi_paths::API_V1_SYNTHTUNNEL_LEASES, request)
            .await
    }

    pub async fn delete_lease(&self, lease_id: &str) -> Result<()> {
        self.transport
            .delete_empty(&openapi_paths::api_v1_synthtunnel_lease(lease_id))
            .await
    }
}
