//! SynthTunnel WebSocket agent — runs in its own tokio runtime,
//! forwarding HTTP requests from the relay to a local server.

use std::collections::HashMap;
use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use futures_util::{SinkExt, StreamExt};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest, http::HeaderValue, Message as WsMessage,
};
use tokio_util::sync::CancellationToken;

use super::errors::TunnelError;
use super::types::SynthTunnelConfig;

// ---------------------------------------------------------------------------
// Protocol types (agent → relay)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(tag = "type")]
#[allow(non_camel_case_types)]
enum AgentOutbound {
    ATTACH {
        agent_id: String,
        leases: Vec<serde_json::Value>,
        capabilities: serde_json::Value,
    },
    RESP_HEADERS {
        lease_id: String,
        rid: String,
        status: u16,
        headers: Vec<(String, String)>,
    },
    RESP_BODY {
        lease_id: String,
        rid: String,
        chunk_b64: String,
        eof: bool,
    },
    RESP_END {
        lease_id: String,
        rid: String,
    },
    RESP_ERROR {
        lease_id: String,
        rid: String,
        code: String,
        message: String,
    },
}

// ---------------------------------------------------------------------------
// Protocol types (relay → agent)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(tag = "type")]
#[allow(non_camel_case_types)]
enum RelayInbound {
    ATTACH_ACK {
        #[allow(dead_code)]
        agent_id: Option<String>,
    },
    REQ_HEADERS {
        lease_id: String,
        rid: String,
        method: Option<String>,
        path: Option<String>,
        query: Option<String>,
        headers: Option<Vec<(String, String)>>,
        #[allow(dead_code)]
        deadline_ms: Option<u64>,
    },
    REQ_BODY {
        rid: String,
        chunk_b64: Option<String>,
    },
    REQ_END {
        rid: String,
    },
    CANCEL {
        rid: String,
    },
}

// ---------------------------------------------------------------------------
// Internal per-request context
// ---------------------------------------------------------------------------

struct RequestContext {
    lease_id: String,
    rid: String,
    method: String,
    path: String,
    query: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Agent handle returned to caller
// ---------------------------------------------------------------------------

type WsTx = Arc<
    Mutex<
        futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            WsMessage,
        >,
    >,
>;

pub struct SynthTunnelAgentHandle {
    cancel: CancellationToken,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl SynthTunnelAgentHandle {
    /// Non-async shutdown signal — safe to call from Python sync code.
    pub fn stop(&self) {
        self.cancel.cancel();
    }

    /// Async shutdown — signals and waits for the background task to finish.
    pub async fn shutdown(&self) {
        self.cancel.cancel();
        let handle = {
            let mut guard = self.join.lock().await;
            guard.take()
        };
        if let Some(h) = handle {
            let _ = h.await;
        }
    }
}

// ---------------------------------------------------------------------------
// HOP-BY-HOP headers to strip
// ---------------------------------------------------------------------------

static HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

// ---------------------------------------------------------------------------
// Client-instance-id helper (mirrors Python's get_client_instance_id)
// ---------------------------------------------------------------------------

fn get_client_instance_id() -> String {
    let home = dirs::home_dir().unwrap_or_default();
    let path = home.join(".synth").join("client_instance_id");
    if let Ok(contents) = std::fs::read_to_string(&path) {
        let val = contents.trim().to_string();
        if val.len() >= 8 {
            return val;
        }
    }
    let new_id = format!("client-{:016x}", rand_u64());
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(&home));
    let _ = std::fs::write(&path, &new_id);
    new_id
}

fn rand_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let s = RandomState::new();
    let mut h = s.build_hasher();
    h.write_u8(0);
    h.finish()
}

// ---------------------------------------------------------------------------
// start_agent — connect, attach, spawn background loop
// ---------------------------------------------------------------------------

pub async fn start_agent(config: SynthTunnelConfig) -> Result<SynthTunnelAgentHandle, TunnelError> {
    // Build WS request with Bearer auth
    let mut request = config
        .ws_url
        .as_str()
        .into_client_request()
        .map_err(|e| TunnelError::websocket(format!("invalid ws url: {e}")))?;
    request.headers_mut().insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {}", config.agent_token))
            .map_err(|e| TunnelError::websocket(format!("bad token header: {e}")))?,
    );

    // Connect
    let (ws_stream, _response) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|e| TunnelError::websocket(format!("ws connect failed: {e}")))?;

    let (ws_sink, mut ws_rx) = ws_stream.split();
    let ws_tx: WsTx = Arc::new(Mutex::new(ws_sink));

    // Send ATTACH
    let attach = AgentOutbound::ATTACH {
        agent_id: get_client_instance_id(),
        leases: vec![serde_json::json!({
            "lease_id": config.lease_id,
            "local_target": {
                "host": config.local_host,
                "port": config.local_port,
            }
        })],
        capabilities: serde_json::json!({
            "streaming": true,
            "max_inflight": config.max_inflight,
            "max_body_chunk_bytes": 65536,
            "supports_cancel": true,
        }),
    };
    send_msg(&ws_tx, &attach).await?;

    // Wait for ATTACH_ACK (10s timeout)
    let ack = timeout(Duration::from_secs(10), ws_rx.next())
        .await
        .map_err(|_| TunnelError::websocket("ATTACH_ACK timeout (10s)"))?
        .ok_or_else(|| TunnelError::websocket("ws closed before ATTACH_ACK"))?
        .map_err(|e| TunnelError::websocket(format!("ws recv error: {e}")))?;

    match ack {
        WsMessage::Text(txt) => {
            let parsed: serde_json::Value = serde_json::from_str(&txt)
                .map_err(|e| TunnelError::websocket(format!("bad ATTACH_ACK json: {e}")))?;
            if parsed.get("type").and_then(|v| v.as_str()) != Some("ATTACH_ACK") {
                return Err(TunnelError::websocket(format!(
                    "expected ATTACH_ACK, got: {}",
                    parsed.get("type").unwrap_or(&serde_json::Value::Null)
                )));
            }
        }
        other => {
            return Err(TunnelError::websocket(format!(
                "expected text ATTACH_ACK, got: {other:?}"
            )));
        }
    }

    // Spawn background loop
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let join = tokio::spawn(agent_loop(config, ws_tx, ws_rx, cancel_clone));

    Ok(SynthTunnelAgentHandle {
        cancel,
        join: Mutex::new(Some(join)),
    })
}

// ---------------------------------------------------------------------------
// agent_loop — main select loop
// ---------------------------------------------------------------------------

type WsRx = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

async fn agent_loop(
    config: SynthTunnelConfig,
    ws_tx: WsTx,
    mut ws_rx: WsRx,
    cancel: CancellationToken,
) {
    let http = HttpClient::new();
    let mut contexts: HashMap<String, RequestContext> = HashMap::new();
    let mut inflight: HashMap<String, tokio::task::JoinHandle<()>> = HashMap::new();

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                break;
            }
            frame = ws_rx.next() => {
                let frame = match frame {
                    Some(Ok(f)) => f,
                    Some(Err(e)) => {
                        eprintln!("[SynthTunnel] ws error: {e}");
                        break;
                    }
                    None => break, // stream ended
                };
                match frame {
                    WsMessage::Text(txt) => {
                        let msg: RelayInbound = match serde_json::from_str(&txt) {
                            Ok(m) => m,
                            Err(e) => {
                                eprintln!("[SynthTunnel] bad frame: {e}");
                                continue;
                            }
                        };
                        match msg {
                            RelayInbound::ATTACH_ACK { .. } => {
                                // already handled during handshake
                            }
                            RelayInbound::REQ_HEADERS {
                                lease_id, rid, method, path, query, headers, ..
                            } => {
                                let hdr_map: HashMap<String, String> = headers
                                    .unwrap_or_default()
                                    .into_iter()
                                    .filter(|(k, _)| !k.is_empty())
                                    .collect();
                                contexts.insert(
                                    rid.clone(),
                                    RequestContext {
                                        lease_id,
                                        rid,
                                        method: method.unwrap_or_else(|| "GET".to_string()),
                                        path: path.unwrap_or_else(|| "/".to_string()),
                                        query: query.unwrap_or_default(),
                                        headers: hdr_map,
                                        body: Vec::new(),
                                    },
                                );
                            }
                            RelayInbound::REQ_BODY { rid, chunk_b64 } => {
                                if let Some(ctx) = contexts.get_mut(&rid) {
                                    if let Some(b64) = chunk_b64 {
                                        if let Ok(bytes) = B64.decode(&b64) {
                                            ctx.body.extend_from_slice(&bytes);
                                        }
                                    }
                                }
                            }
                            RelayInbound::REQ_END { rid } => {
                                if let Some(ctx) = contexts.remove(&rid) {
                                    let tx = ws_tx.clone();
                                    let client = http.clone();
                                    let cfg = config.clone();
                                    let handle = tokio::spawn(async move {
                                        forward_request(ctx, &cfg, tx, client).await;
                                    });
                                    inflight.insert(rid, handle);
                                }
                            }
                            RelayInbound::CANCEL { rid } => {
                                contexts.remove(&rid);
                                if let Some(h) = inflight.remove(&rid) {
                                    h.abort();
                                }
                            }
                        }
                    }
                    WsMessage::Ping(data) => {
                        let _ = ws_tx.lock().await.send(WsMessage::Pong(data)).await;
                    }
                    WsMessage::Close(_) => break,
                    _ => {}
                }
            }
        }
    }

    // Abort any in-flight tasks
    for (_, h) in inflight {
        h.abort();
    }

    // Close WS gracefully
    let _ = ws_tx.lock().await.close().await;
}

// ---------------------------------------------------------------------------
// forward_request — HTTP call to local server, stream response back
// ---------------------------------------------------------------------------

async fn forward_request(
    ctx: RequestContext,
    config: &SynthTunnelConfig,
    ws_tx: WsTx,
    http: HttpClient,
) {
    let mut url = format!(
        "http://{}:{}{}",
        config.local_host, config.local_port, ctx.path
    );
    if !ctx.query.is_empty() {
        url.push('?');
        url.push_str(&ctx.query);
    }

    let method: reqwest::Method = ctx.method.parse().unwrap_or(reqwest::Method::GET);

    // Build headers, stripping hop-by-hop
    let mut headers = reqwest::header::HeaderMap::new();
    for (k, v) in &ctx.headers {
        if HOP_BY_HOP.contains(&k.to_lowercase().as_str()) {
            continue;
        }
        if let (Ok(name), Ok(val)) = (
            reqwest::header::HeaderName::from_bytes(k.as_bytes()),
            reqwest::header::HeaderValue::from_str(v),
        ) {
            headers.insert(name, val);
        }
    }

    // Inject local API keys — always overwrite relay-supplied auth headers
    // because the relay forwards the worker_token, not the local env key.
    if !config.local_api_keys.is_empty() {
        let primary = &config.local_api_keys[0];
        if let Ok(v) = reqwest::header::HeaderValue::from_str(primary) {
            headers.insert("x-api-key", v);
        }
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {primary}")) {
            headers.insert("authorization", v);
        }
        if config.local_api_keys.len() > 1 {
            let joined = config.local_api_keys.join(",");
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&joined) {
                headers.insert("x-api-keys", v);
            }
        }
    }

    // Inject tunnel metadata headers
    if !headers.contains_key("x-synthtunnel-lease-id") {
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&ctx.lease_id) {
            headers.insert("x-synthtunnel-lease-id", v);
        }
    }
    if !headers.contains_key("x-synthtunnel-request-id") {
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&ctx.rid) {
            headers.insert("x-synthtunnel-request-id", v);
        }
    }
    if !headers.contains_key("x-forwarded-proto") {
        headers.insert(
            "x-forwarded-proto",
            reqwest::header::HeaderValue::from_static("https"),
        );
    }

    let rid = ctx.rid.clone();
    let lease_id = ctx.lease_id.clone();

    let result = http
        .request(method, &url)
        .headers(headers)
        .body(ctx.body)
        .send()
        .await;

    match result {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let resp_headers: Vec<(String, String)> = resp
                .headers()
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                .collect();

            // Send RESP_HEADERS
            let _ = send_msg(
                &ws_tx,
                &AgentOutbound::RESP_HEADERS {
                    lease_id: lease_id.clone(),
                    rid: rid.clone(),
                    status,
                    headers: resp_headers,
                },
            )
            .await;

            // Stream body chunks
            let mut stream = resp.bytes_stream();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) if !chunk.is_empty() => {
                        let _ = send_msg(
                            &ws_tx,
                            &AgentOutbound::RESP_BODY {
                                lease_id: lease_id.clone(),
                                rid: rid.clone(),
                                chunk_b64: B64.encode(&chunk),
                                eof: false,
                            },
                        )
                        .await;
                    }
                    Err(e) => {
                        let _ = send_msg(
                            &ws_tx,
                            &AgentOutbound::RESP_ERROR {
                                lease_id: lease_id.clone(),
                                rid: rid.clone(),
                                code: "LOCAL_BAD_RESPONSE".to_string(),
                                message: e.to_string(),
                            },
                        )
                        .await;
                        return;
                    }
                    _ => {}
                }
            }

            // Send RESP_END
            let _ = send_msg(&ws_tx, &AgentOutbound::RESP_END { lease_id, rid }).await;
        }
        Err(e) => {
            let _ = send_msg(
                &ws_tx,
                &AgentOutbound::RESP_ERROR {
                    lease_id,
                    rid,
                    code: "LOCAL_CONNECT_FAILED".to_string(),
                    message: e.to_string(),
                },
            )
            .await;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: send a JSON text frame over the WS
// ---------------------------------------------------------------------------

async fn send_msg(ws_tx: &WsTx, msg: &AgentOutbound) -> Result<(), TunnelError> {
    let json = serde_json::to_string(msg)
        .map_err(|e| TunnelError::websocket(format!("serialize error: {e}")))?;
    ws_tx
        .lock()
        .await
        .send(WsMessage::Text(json))
        .await
        .map_err(|e| TunnelError::websocket(format!("ws send error: {e}")))?;
    Ok(())
}
