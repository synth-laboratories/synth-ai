use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::shared_client::DEFAULT_CONNECT_TIMEOUT_SECS;
use crate::tunnels::errors::TunnelError;
use crate::tunnels::ports::{is_port_available, kill_port};
use crate::tunnels::types::{GatewayState, GatewayStatus};

pub struct TunnelGateway {
    port: u16,
    routes: Arc<RwLock<HashMap<String, (String, u16)>>>,
    state: GatewayState,
    server_task: Option<JoinHandle<()>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    error: Option<String>,
}

impl TunnelGateway {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            routes: Arc::new(RwLock::new(HashMap::new())),
            state: GatewayState::Stopped,
            server_task: None,
            shutdown_tx: None,
            error: None,
        }
    }

    pub fn status(&self) -> GatewayStatus {
        let routes = self
            .routes
            .read()
            .iter()
            .map(|(k, (h, p))| (k.clone(), h.clone(), *p))
            .collect();
        GatewayStatus {
            state: self.state.clone(),
            port: self.port,
            routes,
            error: self.error.clone(),
        }
    }

    pub fn is_running(&self) -> bool {
        self.state == GatewayState::Running
    }

    pub fn add_route(&self, prefix: &str, host: &str, port: u16) {
        self.routes
            .write()
            .insert(prefix.to_string(), (host.to_string(), port));
    }

    pub fn remove_route(&self, prefix: &str) -> bool {
        self.routes.write().remove(prefix).is_some()
    }

    pub async fn start(&mut self, force: bool) -> Result<(), TunnelError> {
        if self.state == GatewayState::Running {
            return Ok(());
        }
        self.state = GatewayState::Starting;
        if !is_port_available(self.port, "127.0.0.1") {
            if force {
                let _ = kill_port(self.port);
                tokio::time::sleep(Duration::from_millis(500)).await;
            } else {
                self.state = GatewayState::Error;
                self.error = Some(format!("port {} in use", self.port));
                return Err(TunnelError::gateway(format!(
                    "gateway port {} in use",
                    self.port
                )));
            }
        }
        let listener = TcpListener::bind(("127.0.0.1", self.port))
            .await
            .map_err(|e| TunnelError::gateway(e.to_string()))?;
        let routes = self.routes.clone();
        let (tx, rx) = oneshot::channel::<()>();
        self.shutdown_tx = Some(tx);
        let task = tokio::spawn(async move {
            use hyper_util::rt::TokioIo;
            let mut shutdown = rx;
            loop {
                tokio::select! {
                    _ = &mut shutdown => {
                        break;
                    }
                    accept = listener.accept() => {
                        let (stream, _) = match accept {
                            Ok(pair) => pair,
                            Err(_) => continue,
                        };
                        let routes = routes.clone();
                        tokio::spawn(async move {
                            let service = service_fn(move |req| handle_req(req, routes.clone()));
                            let _ = hyper::server::conn::http1::Builder::new()
                                .serve_connection(TokioIo::new(stream), service)
                                .await;
                        });
                    }
                }
            }
        });
        self.server_task = Some(task);
        self.state = GatewayState::Running;
        Ok(())
    }

    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(task) = self.server_task.take() {
            task.abort();
        }
        self.routes.write().clear();
        self.state = GatewayState::Stopped;
    }
}

async fn handle_req(
    req: Request<Incoming>,
    routes: Arc<RwLock<HashMap<String, (String, u16)>>>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let path = req.uri().path().to_string();
    if path == "/__synth/gateway/health" {
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from_static(
                b"{\"status\":\"ok\",\"gateway\":\"running\"}",
            )))
            .unwrap());
    }

    if path.ends_with("/__synth/ready") {
        let prefix = path.trim_end_matches("/__synth/ready");
        // Extract the route info and drop the lock guard before await
        let route_info = routes.read().get(prefix).cloned();
        if let Some((host, port)) = route_info {
            let url = format!("http://{host}:{port}/");
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .pool_max_idle_per_host(20)
                .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
                .no_proxy()
                .build()
                .unwrap();
            let resp = client.get(url).send().await;
            if let Ok(resp) = resp {
                if resp.status().as_u16() < 500 {
                    let body = format!(
                        "{{\"status\":\"ok\",\"route\":\"{prefix}\",\"target\":\"{host}:{port}\"}}"
                    );
                    return Ok(Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(body)))
                        .unwrap());
                }
            }
            return Ok(Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from_static(
                    b"{\"status\":\"unavailable\",\"error\":\"target_unreachable\"}",
                )))
                .unwrap());
        }
        return Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from_static(
                b"{\"status\":\"not_found\",\"error\":\"route_not_found\"}",
            )))
            .unwrap());
    }

    let target = {
        let map = routes.read();
        map.iter()
            .find(|(prefix, _)| path.starts_with(prefix.as_str()))
            .map(|(prefix, (host, port))| {
                let mut stripped = path[prefix.len()..].to_string();
                if stripped.is_empty() {
                    stripped = "/".to_string();
                } else if !stripped.starts_with('/') {
                    stripped = format!("/{}", stripped);
                }
                (host.clone(), *port, stripped)
            })
    };

    let (host, port, stripped) = match target {
        Some(t) => t,
        None => {
            return Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from_static(
                    b"{\"error\":\"no_route\",\"message\":\"No route found\"}",
                )))
                .unwrap());
        }
    };

    let mut target_url = format!("http://{host}:{port}{stripped}");
    if let Some(q) = req.uri().query() {
        target_url = format!("{target_url}?{q}");
    }

    let method = req.method().clone();
    // Capture headers before consuming the body
    let req_headers: Vec<_> = req
        .headers()
        .iter()
        .filter(|(k, _)| !is_hop_by_hop(k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let body = req.into_body().collect().await?.to_bytes();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .pool_max_idle_per_host(50)
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .build()
        .unwrap();
    let mut builder = client.request(method, target_url);
    for (k, v) in req_headers {
        builder = builder.header(k, v);
    }
    let resp = builder.body(body.to_vec()).send().await;
    match resp {
        Ok(resp) => {
            let status = resp.status();
            // Capture headers before consuming the body
            let resp_headers: Vec<_> = resp
                .headers()
                .iter()
                .filter(|(k, _)| {
                    k.as_str() != "content-length" && k.as_str() != "transfer-encoding"
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let bytes = resp.bytes().await.unwrap_or_default();
            let mut response = Response::builder().status(status);
            for (k, v) in resp_headers {
                response = response.header(k, v);
            }
            Ok(response.body(Full::new(Bytes::from(bytes))).unwrap())
        }
        Err(_) => Ok(Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from_static(
                b"{\"error\":\"bad_gateway\"}",
            )))
            .unwrap()),
    }
}

fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

static GATEWAY: Lazy<Mutex<TunnelGateway>> = Lazy::new(|| Mutex::new(TunnelGateway::new(8016)));

pub fn get_gateway() -> &'static Mutex<TunnelGateway> {
    &GATEWAY
}
