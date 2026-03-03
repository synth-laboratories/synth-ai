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
    let (ws_tx, ws_rx) = connect_and_attach(&config).await?;

    // Spawn background supervisor loop with reconnect handling.
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let join = tokio::spawn(agent_supervisor_loop(config, ws_tx, ws_rx, cancel_clone));

    Ok(SynthTunnelAgentHandle {
        cancel,
        join: Mutex::new(Some(join)),
    })
}

async fn connect_and_attach(config: &SynthTunnelConfig) -> Result<(WsTx, WsRx), TunnelError> {
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

    Ok((ws_tx, ws_rx))
}

// ---------------------------------------------------------------------------
// agent_loop — main select loop
// ---------------------------------------------------------------------------

type WsRx = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

async fn wait_or_cancel(cancel: &CancellationToken, delay: Duration) -> bool {
    tokio::select! {
        _ = cancel.cancelled() => true,
        _ = tokio::time::sleep(delay) => false,
    }
}

async fn agent_supervisor_loop(
    config: SynthTunnelConfig,
    initial_ws_tx: WsTx,
    initial_ws_rx: WsRx,
    cancel: CancellationToken,
) {
    let mut reconnect_delay = Duration::from_millis(250);
    let mut first_ws_tx = Some(initial_ws_tx);
    let mut first_ws_rx = Some(initial_ws_rx);

    loop {
        if cancel.is_cancelled() {
            break;
        }

        let (ws_tx, ws_rx) =
            if let (Some(ws_tx), Some(ws_rx)) = (first_ws_tx.take(), first_ws_rx.take()) {
                (ws_tx, ws_rx)
            } else {
                match connect_and_attach(&config).await {
                    Ok((ws_tx, ws_rx)) => (ws_tx, ws_rx),
                    Err(e) => {
                        eprintln!("[SynthTunnel] ws reconnect failed: {e}");
                        if wait_or_cancel(&cancel, reconnect_delay).await {
                            break;
                        }
                        reconnect_delay = reconnect_delay
                            .saturating_mul(2)
                            .min(Duration::from_secs(5));
                        continue;
                    }
                }
            };

        reconnect_delay = Duration::from_millis(250);
        agent_loop_connection(config.clone(), ws_tx, ws_rx, cancel.clone()).await;
        if cancel.is_cancelled() {
            break;
        }

        eprintln!("[SynthTunnel] ws disconnected; reconnecting");
        if wait_or_cancel(&cancel, reconnect_delay).await {
            break;
        }
    }
}

async fn agent_loop_connection(
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

    // Relay-signed container auth is forwarded by the tunnel backend.

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

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio::time::{timeout, Duration, Instant};
    use tokio_tungstenite::tungstenite::Message;

    type TestWs = tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>;

    fn start_local_http_server(expected_requests: usize) -> (u16, thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind local http");
        let port = listener.local_addr().expect("local http addr").port();
        let handle = thread::spawn(move || {
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("accept local http");
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                    .expect("set read timeout");
                let mut buf = [0_u8; 4096];
                let read_len = stream.read(&mut buf).unwrap_or(0);
                let request_text = String::from_utf8_lossy(&buf[..read_len]);
                let request_line = request_text.lines().next().unwrap_or("GET / HTTP/1.1");
                let request_path = request_line.split_whitespace().nth(1).unwrap_or("/");
                let body = format!(r#"{{"ok":true,"path":"{request_path}"}}"#);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write local http response");
                stream.flush().expect("flush local http response");
            }
        });
        (port, handle)
    }

    async fn recv_text_json(ws: &mut TestWs) -> serde_json::Value {
        loop {
            let frame = timeout(Duration::from_secs(5), ws.next())
                .await
                .expect("ws frame timeout")
                .expect("ws stream ended")
                .expect("ws recv error");
            if let Message::Text(txt) = frame {
                return serde_json::from_str(&txt).expect("parse ws json");
            }
        }
    }

    async fn send_request(ws: &mut TestWs, rid: &str, path: &str, query: &str) {
        ws.send(Message::Text(
            serde_json::json!({
                "type": "REQ_HEADERS",
                "lease_id": "lease-1",
                "rid": rid,
                "method": "GET",
                "path": path,
                "query": query,
                "headers": [["accept", "application/json"]],
                "deadline_ms": 10_000
            })
            .to_string(),
        ))
        .await
        .expect("send REQ_HEADERS");
        ws.send(Message::Text(
            serde_json::json!({
                "type": "REQ_END",
                "rid": rid
            })
            .to_string(),
        ))
        .await
        .expect("send REQ_END");
    }

    async fn collect_response(ws: &mut TestWs, rid: &str) -> (u16, String) {
        let mut status = 0_u16;
        let mut body = Vec::<u8>::new();
        loop {
            let msg = recv_text_json(ws).await;
            if msg.get("rid").and_then(|v| v.as_str()) != Some(rid) {
                continue;
            }
            match msg.get("type").and_then(|v| v.as_str()) {
                Some("RESP_HEADERS") => {
                    status = msg
                        .get("status")
                        .and_then(|v| v.as_u64())
                        .and_then(|v| u16::try_from(v).ok())
                        .unwrap_or(0);
                }
                Some("RESP_BODY") => {
                    let chunk_b64 = msg.get("chunk_b64").and_then(|v| v.as_str()).unwrap_or("");
                    if !chunk_b64.is_empty() {
                        let chunk = B64.decode(chunk_b64).expect("decode chunk");
                        body.extend_from_slice(&chunk);
                    }
                }
                Some("RESP_ERROR") => {
                    panic!("relay received RESP_ERROR: {msg}");
                }
                Some("RESP_END") => {
                    break;
                }
                _ => {}
            }
        }
        (status, String::from_utf8(body).expect("utf8 response body"))
    }

    async fn collect_error_code(ws: &mut TestWs, rid: &str) -> String {
        loop {
            let msg = recv_text_json(ws).await;
            if msg.get("rid").and_then(|v| v.as_str()) != Some(rid) {
                continue;
            }
            if msg.get("type").and_then(|v| v.as_str()) == Some("RESP_ERROR") {
                return msg
                    .get("code")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn forwards_request_response_over_ws_relay() {
        let (local_port, local_http_thread) = start_local_http_server(1);
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ws listener");
        let addr = listener.local_addr().expect("ws listener addr");

        let relay = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept relay ws");
            let mut ws = tokio_tungstenite::accept_async(stream)
                .await
                .expect("accept relay websocket");
            let attach = recv_text_json(&mut ws).await;
            assert_eq!(attach.get("type").and_then(|v| v.as_str()), Some("ATTACH"));
            ws.send(Message::Text(
                serde_json::json!({"type":"ATTACH_ACK"}).to_string(),
            ))
            .await
            .expect("send ATTACH_ACK");

            send_request(&mut ws, "rid-forward", "/probe", "seed=1").await;
            collect_response(&mut ws, "rid-forward").await
        });

        let config = SynthTunnelConfig {
            ws_url: format!("ws://{}/agent", addr),
            agent_token: "agent-token".to_string(),
            lease_id: "lease-1".to_string(),
            local_host: "127.0.0.1".to_string(),
            local_port,
            public_url: "https://st.usesynth.ai/s/rt_test".to_string(),
            worker_token: "worker-token".to_string(),
            container_keys: vec!["env-key".to_string()],
            max_inflight: 16,
        };

        let handle = start_agent(config).await.expect("start agent");
        let (status, body) = timeout(Duration::from_secs(10), relay)
            .await
            .expect("relay timeout")
            .expect("relay join");
        handle.shutdown().await;
        local_http_thread.join().expect("join local http");

        assert_eq!(status, 200);
        assert!(
            body.contains("/probe?seed=1"),
            "expected forwarded query path in body, got: {body}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconnects_after_ungraceful_ws_disconnect() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test ws listener");
        let addr = listener.local_addr().expect("listener local addr");
        let connections = Arc::new(AtomicUsize::new(0));
        let connections_for_server = Arc::clone(&connections);

        let server = tokio::spawn(async move {
            while connections_for_server.load(Ordering::SeqCst) < 2 {
                let (stream, _) = listener.accept().await.expect("accept");
                let idx = connections_for_server.fetch_add(1, Ordering::SeqCst) + 1;
                tokio::spawn(async move {
                    let mut ws = tokio_tungstenite::accept_async(stream)
                        .await
                        .expect("ws accept");
                    let _ = ws.next().await;
                    let _ = ws
                        .send(Message::Text(
                            serde_json::json!({"type":"ATTACH_ACK"}).to_string(),
                        ))
                        .await;
                    if idx == 1 {
                        // Drop the socket without sending CLOSE to simulate abrupt resets.
                        drop(ws);
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    let _ = ws.close(None).await;
                });
            }
        });

        let config = SynthTunnelConfig {
            ws_url: format!("ws://{}/agent", addr),
            agent_token: "agent-token".to_string(),
            lease_id: "lease-1".to_string(),
            local_host: "127.0.0.1".to_string(),
            local_port: 9876,
            public_url: "https://st.usesynth.ai/s/rt_test".to_string(),
            worker_token: "worker-token".to_string(),
            container_keys: vec!["env-key".to_string()],
            max_inflight: 16,
        };

        let handle = start_agent(config).await.expect("start agent");

        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if connections.load(Ordering::SeqCst) >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        handle.shutdown().await;
        let _ = server.await;
        assert!(
            connections.load(Ordering::SeqCst) >= 2,
            "expected reconnect after abrupt disconnect"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn emits_local_connect_failed_when_upstream_unreachable() {
        let dead_local_port = 65_431;
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind relay ws listener");
        let addr = listener.local_addr().expect("relay addr");

        let relay = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept relay ws");
            let mut ws = tokio_tungstenite::accept_async(stream)
                .await
                .expect("accept relay websocket");
            let attach = recv_text_json(&mut ws).await;
            assert_eq!(attach.get("type").and_then(|v| v.as_str()), Some("ATTACH"));
            ws.send(Message::Text(
                serde_json::json!({"type":"ATTACH_ACK"}).to_string(),
            ))
            .await
            .expect("send ATTACH_ACK");

            send_request(&mut ws, "rid-connect-fail", "/payload", "").await;
            collect_error_code(&mut ws, "rid-connect-fail").await
        });

        let config = SynthTunnelConfig {
            ws_url: format!("ws://{}/agent", addr),
            agent_token: "agent-token".to_string(),
            lease_id: "lease-1".to_string(),
            local_host: "127.0.0.1".to_string(),
            local_port: dead_local_port,
            public_url: "https://st.usesynth.ai/s/rt_test".to_string(),
            worker_token: "worker-token".to_string(),
            container_keys: vec!["env-key".to_string()],
            max_inflight: 16,
        };

        let handle = start_agent(config).await.expect("start agent");
        let error_code = timeout(Duration::from_secs(10), relay)
            .await
            .expect("relay timeout")
            .expect("relay join");
        handle.shutdown().await;

        assert_eq!(error_code, "LOCAL_CONNECT_FAILED");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconnects_and_forwards_request_after_reset() {
        let (local_port, local_http_thread) = start_local_http_server(1);
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind relay ws listener");
        let addr = listener.local_addr().expect("relay addr");

        let relay = tokio::spawn(async move {
            let (stream1, _) = listener.accept().await.expect("accept relay ws #1");
            let mut ws1 = tokio_tungstenite::accept_async(stream1)
                .await
                .expect("accept relay websocket #1");
            let attach1 = recv_text_json(&mut ws1).await;
            assert_eq!(attach1.get("type").and_then(|v| v.as_str()), Some("ATTACH"));
            ws1.send(Message::Text(
                serde_json::json!({"type":"ATTACH_ACK"}).to_string(),
            ))
            .await
            .expect("send ATTACH_ACK #1");
            drop(ws1);

            let (stream2, _) = listener.accept().await.expect("accept relay ws #2");
            let mut ws2 = tokio_tungstenite::accept_async(stream2)
                .await
                .expect("accept relay websocket #2");
            let attach2 = recv_text_json(&mut ws2).await;
            assert_eq!(attach2.get("type").and_then(|v| v.as_str()), Some("ATTACH"));
            ws2.send(Message::Text(
                serde_json::json!({"type":"ATTACH_ACK"}).to_string(),
            ))
            .await
            .expect("send ATTACH_ACK #2");

            send_request(&mut ws2, "rid-reconnect", "/probe", "seed=2").await;
            collect_response(&mut ws2, "rid-reconnect").await
        });

        let config = SynthTunnelConfig {
            ws_url: format!("ws://{}/agent", addr),
            agent_token: "agent-token".to_string(),
            lease_id: "lease-1".to_string(),
            local_host: "127.0.0.1".to_string(),
            local_port,
            public_url: "https://st.usesynth.ai/s/rt_test".to_string(),
            worker_token: "worker-token".to_string(),
            container_keys: vec!["env-key".to_string()],
            max_inflight: 16,
        };

        let handle = start_agent(config).await.expect("start agent");
        let (status, body) = timeout(Duration::from_secs(10), relay)
            .await
            .expect("relay timeout")
            .expect("relay join");
        handle.shutdown().await;
        local_http_thread.join().expect("join local http");

        assert_eq!(status, 200);
        assert!(
            body.contains("/probe?seed=2"),
            "expected forwarded query path in body after reconnect, got: {body}"
        );
    }
}
