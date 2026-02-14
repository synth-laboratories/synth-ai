//! SynthTunnel lease API client.
//!
//! This module owns the HTTP interaction for creating/closing SynthTunnel leases.
//! The WebSocket agent implementation lives in `tunnels::synth_tunnel`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::CoreError;
use crate::http::{HttpClient, HttpError};
use crate::urls::normalize_backend_base;

const SYNTH_TUNNEL_LEASES_ENDPOINT: &str = "/api/v1/synthtunnel/leases";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthTunnelLease {
    pub lease_id: String,
    pub route_token: String,
    pub public_base_url: String,
    pub public_url: String,
    pub agent_url: String,
    pub agent_token: String,
    pub worker_token: String,
    pub expires_at: String,
    #[serde(default)]
    pub limits: Value,
    #[serde(default)]
    pub heartbeat: Value,
}

#[derive(Clone)]
pub struct SynthTunnelLeaseClient {
    http: HttpClient,
    backend_base: String,
}

impl SynthTunnelLeaseClient {
    pub fn new(api_key: &str, backend_url: &str, timeout_secs: u64) -> Result<Self, CoreError> {
        let backend_base = normalize_backend_base(backend_url)?.to_string();
        let http = HttpClient::new(&backend_base, api_key, timeout_secs)
            .map_err(map_http_error)?;
        Ok(Self { http, backend_base })
    }

    pub fn backend_base(&self) -> &str {
        &self.backend_base
    }

    pub async fn create_lease(
        &self,
        client_instance_id: &str,
        local_host: &str,
        local_port: u16,
        requested_ttl_seconds: i64,
        metadata: Option<Value>,
        capabilities: Option<Value>,
    ) -> Result<SynthTunnelLease, CoreError> {
        #[derive(Deserialize)]
        struct AgentConnect {
            url: String,
            agent_token: String,
        }

        #[derive(Deserialize)]
        struct WireLeaseResponse {
            lease_id: String,
            route_token: String,
            public_base_url: String,
            public_url: String,
            agent_connect: AgentConnect,
            worker_token: String,
            expires_at: String,
            #[serde(default)]
            limits: Value,
            #[serde(default)]
            heartbeat: Value,
        }

        let payload = serde_json::json!({
            "client_instance_id": client_instance_id,
            "local_target": {"host": local_host, "port": local_port},
            "requested_ttl_seconds": requested_ttl_seconds,
            "metadata": metadata.unwrap_or_else(|| serde_json::json!({})),
            "capabilities": capabilities.unwrap_or_else(|| serde_json::json!({})),
        });

        let wire: WireLeaseResponse = self
            .http
            .post_json(SYNTH_TUNNEL_LEASES_ENDPOINT, &payload)
            .await
            .map_err(map_http_error)?;

        Ok(SynthTunnelLease {
            lease_id: wire.lease_id,
            route_token: wire.route_token,
            public_base_url: wire.public_base_url,
            public_url: wire.public_url,
            agent_url: wire.agent_connect.url,
            agent_token: wire.agent_connect.agent_token,
            worker_token: wire.worker_token,
            expires_at: wire.expires_at,
            limits: wire.limits,
            heartbeat: wire.heartbeat,
        })
    }

    pub async fn close_lease(&self, lease_id: &str) -> Result<(), CoreError> {
        let path = format!("{}/{}", SYNTH_TUNNEL_LEASES_ENDPOINT, lease_id);
        self.http.delete(&path).await.map_err(map_http_error)
    }
}

fn map_http_error(e: HttpError) -> CoreError {
    match e {
        HttpError::Response(detail) => {
            if detail.status == 401 || detail.status == 403 {
                CoreError::Authentication(format!("authentication failed: {}", detail))
            } else if detail.status == 429 {
                CoreError::UsageLimit(crate::UsageLimitInfo::from_http_429("synthtunnel", &detail))
            } else {
                CoreError::HttpResponse(crate::HttpErrorInfo {
                    status: detail.status,
                    url: detail.url,
                    message: detail.message,
                    body_snippet: detail.body_snippet,
                })
            }
        }
        HttpError::Request(e) => CoreError::Http(e),
        other => CoreError::Internal(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;

    #[test]
    fn test_endpoint_const() {
        assert_eq!(SYNTH_TUNNEL_LEASES_ENDPOINT, "/api/v1/synthtunnel/leases");
    }

    #[tokio::test]
    async fn test_create_and_close_lease() {
        let server = MockServer::start();

        let create = server.mock(|when, then| {
            when.method(POST)
                .path("/api/v1/synthtunnel/leases")
                .json_body(json!({
                    "client_instance_id": "client-1",
                    "local_target": {"host": "127.0.0.1", "port": 8001},
                    "requested_ttl_seconds": 3600,
                    "metadata": {},
                    "capabilities": {},
                }));
            then.status(200).json_body(json!({
                "lease_id": "lease_1",
                "route_token": "rt_1",
                "public_base_url": format!("{}/s", server.base_url()),
                "public_url": format!("{}/s/rt_1", server.base_url()),
                "agent_connect": {"url": "wss://agent.example/ws", "agent_token": "agent_tok"},
                "worker_token": "worker_tok",
                "expires_at": "2026-02-14T00:00:00Z",
                "limits": {"max_inflight": 128},
                "heartbeat": {},
            }));
        });

        let close = server.mock(|when, then| {
            when.method(DELETE)
                .path("/api/v1/synthtunnel/leases/lease_1");
            then.status(204);
        });

        let client = SynthTunnelLeaseClient::new("sk_test", &server.base_url(), 5).unwrap();
        let lease = client
            .create_lease(
                "client-1",
                "127.0.0.1",
                8001,
                3600,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(lease.lease_id, "lease_1");
        assert_eq!(lease.worker_token, "worker_tok");
        assert_eq!(lease.limits["max_inflight"], 128);

        client.close_lease(&lease.lease_id).await.unwrap();

        create.assert_hits(1);
        close.assert_hits(1);
    }
}
