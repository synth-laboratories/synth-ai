use chrono::{DateTime, Utc};
use reqwest::StatusCode;
use serde::Deserialize;
use std::time::Duration;

use crate::shared_client::DEFAULT_CONNECT_TIMEOUT_SECS;
use crate::tunnels::errors::TunnelError;
use crate::tunnels::types::{LeaseInfo, LeaseState};

#[derive(Clone)]
pub struct LeaseClient {
    api_key: String,
    backend_url: String,
    client: reqwest::Client,
}

impl LeaseClient {
    pub fn new(api_key: String, backend_url: String, timeout_s: u64) -> Result<Self, TunnelError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_s))
            .pool_max_idle_per_host(20)
            .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
            .user_agent("synth-core/0.1")
            .build()
            .map_err(|e| TunnelError::api(e.to_string()))?;
        Ok(Self {
            api_key,
            backend_url: backend_url.trim_end_matches('/').to_string(),
            client,
        })
    }

    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<serde_json::Value>,
        params: Option<Vec<(String, String)>>,
    ) -> Result<serde_json::Value, TunnelError> {
        let url = format!("{}/api/v1/tunnels{}", self.backend_url, path);
        let mut req = self
            .client
            .request(method, url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");
        if let Some(b) = body {
            req = req.json(&b);
        }
        if let Some(p) = params {
            req = req.query(&p);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| TunnelError::api(e.to_string()))?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Err(TunnelError::lease("lease not found"));
        }
        if resp.status().is_client_error() || resp.status().is_server_error() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(TunnelError::api(format!(
                "backend error status={} body={}",
                status, text
            )));
        }
        let json = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|e| TunnelError::api(e.to_string()))?;
        Ok(json)
    }

    pub async fn create_lease(
        &self,
        client_instance_id: &str,
        local_host: &str,
        local_port: u16,
        app_name: Option<&str>,
        requested_ttl_seconds: i64,
        reuse_connector: bool,
        idempotency_key: Option<&str>,
    ) -> Result<LeaseInfo, TunnelError> {
        let body = serde_json::json!({
            "client_instance_id": client_instance_id,
            "local_host": local_host,
            "local_port": local_port,
            "app_name": app_name,
            "requested_ttl_seconds": requested_ttl_seconds,
            "reuse_connector": reuse_connector,
            "idempotency_key": idempotency_key,
        });
        let data = self
            .request(reqwest::Method::POST, "/lease", Some(body), None)
            .await?;

        #[derive(Deserialize)]
        struct LeasePayload {
            lease_id: String,
            managed_tunnel_id: String,
            hostname: String,
            route_prefix: String,
            public_url: String,
            expires_at: String,
            tunnel_token: String,
            access_client_id: Option<String>,
            access_client_secret: Option<String>,
            gateway_port: Option<u16>,
            diagnostics_hint: Option<String>,
        }

        let payload: LeasePayload = serde_json::from_value(data)
            .map_err(|e| TunnelError::api(format!("invalid lease payload: {e}")))?;
        let expires_at = DateTime::parse_from_rfc3339(&payload.expires_at)
            .map_err(|e| TunnelError::api(format!("invalid expires_at: {e}")))?
            .with_timezone(&Utc);
        Ok(LeaseInfo {
            lease_id: payload.lease_id,
            managed_tunnel_id: payload.managed_tunnel_id,
            hostname: payload.hostname,
            route_prefix: payload.route_prefix,
            public_url: payload.public_url,
            local_host: local_host.to_string(),
            local_port,
            expires_at,
            tunnel_token: payload.tunnel_token,
            access_client_id: payload.access_client_id,
            access_client_secret: payload.access_client_secret,
            gateway_port: payload.gateway_port.unwrap_or(8016),
            state: LeaseState::Pending,
            diagnostics_hint: payload.diagnostics_hint.unwrap_or_default(),
        })
    }

    pub async fn heartbeat(
        &self,
        lease_id: &str,
        connected_to_edge: bool,
        gateway_ready: bool,
        local_ready: bool,
        last_error: Option<&str>,
    ) -> Result<(String, i64), TunnelError> {
        let body = serde_json::json!({
            "connected_to_edge": connected_to_edge,
            "gateway_ready": gateway_ready,
            "local_ready": local_ready,
            "last_error": last_error.map(|e| e.chars().take(1000).collect::<String>()),
        });
        let data = self
            .request(
                reqwest::Method::POST,
                &format!("/lease/{lease_id}/heartbeat"),
                Some(body),
                None,
            )
            .await?;
        let action = data
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("none")
            .to_string();
        let next = data
            .get("next_heartbeat_seconds")
            .and_then(|v| v.as_i64())
            .unwrap_or(30);
        Ok((action, next))
    }

    pub async fn release(&self, lease_id: &str) -> Result<(), TunnelError> {
        let _ = self
            .request(
                reqwest::Method::POST,
                &format!("/lease/{lease_id}/release"),
                None,
                None,
            )
            .await?;
        Ok(())
    }

    pub async fn list_leases(
        &self,
        client_instance_id: Option<&str>,
        include_expired: bool,
    ) -> Result<Vec<LeaseInfo>, TunnelError> {
        let mut params: Vec<(String, String)> = Vec::new();
        if let Some(id) = client_instance_id {
            params.push(("client_instance_id".to_string(), id.to_string()));
        }
        if include_expired {
            params.push(("include_expired".to_string(), "true".to_string()));
        }

        let data = self
            .request(reqwest::Method::GET, "/lease", None, Some(params))
            .await?;

        let items = data
            .as_array()
            .ok_or_else(|| TunnelError::api("invalid lease list payload".to_string()))?;

        #[derive(Deserialize)]
        struct LeaseListItem {
            lease_id: String,
            managed_tunnel_id: String,
            hostname: String,
            route_prefix: String,
            public_url: String,
            expires_at: String,
            gateway_port: Option<u16>,
            diagnostics_hint: Option<String>,
            state: Option<String>,
        }

        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let payload: LeaseListItem = serde_json::from_value(item.clone())
                .map_err(|e| TunnelError::api(format!("invalid lease list entry: {e}")))?;
            let expires_at = DateTime::parse_from_rfc3339(&payload.expires_at)
                .map_err(|e| TunnelError::api(format!("invalid expires_at: {e}")))?
                .with_timezone(&Utc);
            let state = match payload.state.as_deref() {
                Some("expired") => LeaseState::Expired,
                Some("released") => LeaseState::Released,
                Some("failed") => LeaseState::Failed,
                Some("pending") => LeaseState::Pending,
                _ => LeaseState::Active,
            };
            out.push(LeaseInfo {
                lease_id: payload.lease_id,
                managed_tunnel_id: payload.managed_tunnel_id,
                hostname: payload.hostname,
                route_prefix: payload.route_prefix,
                public_url: payload.public_url,
                local_host: "127.0.0.1".to_string(),
                local_port: 0,
                expires_at,
                tunnel_token: "".to_string(),
                access_client_id: None,
                access_client_secret: None,
                gateway_port: payload.gateway_port.unwrap_or(8016),
                state,
                diagnostics_hint: payload.diagnostics_hint.unwrap_or_default(),
            });
        }

        Ok(out)
    }
}
