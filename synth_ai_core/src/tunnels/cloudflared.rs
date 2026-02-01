use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;
use tokio::task::JoinHandle;

use crate::shared_client::DEFAULT_CONNECT_TIMEOUT_SECS;
use crate::tunnels::errors::TunnelError;

static URL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"https://[a-z0-9-]+\\.trycloudflare\\.com").unwrap());

const CLOUDFLARED_RELEASES: &str = "https://updatecloudflared.com/launcher";

#[derive(Debug)]
pub struct ManagedProcess {
    pub child: Child,
    pub logs: Arc<Mutex<VecDeque<String>>>,
    stdout_task: Option<JoinHandle<()>>,
    stderr_task: Option<JoinHandle<()>>,
}

impl ManagedProcess {
    async fn stop(&mut self) {
        let _ = self.child.start_kill();
        let _ = self.child.wait().await;
        if let Some(task) = self.stdout_task.take() {
            task.abort();
        }
        if let Some(task) = self.stderr_task.take() {
            task.abort();
        }
    }
}

static TRACKED: Lazy<Mutex<HashMap<usize, ManagedProcess>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

pub fn track_process(proc: ManagedProcess) -> usize {
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    TRACKED.lock().insert(id, proc);
    id
}

pub async fn stop_tracked(id: usize) -> Result<(), TunnelError> {
    let mut guard = TRACKED.lock();
    if let Some(mut proc) = guard.remove(&id) {
        proc.stop().await;
        return Ok(());
    }
    Err(TunnelError::process(format!("process id {id} not found")))
}

pub async fn cleanup_all() {
    let mut procs = TRACKED.lock();
    for (_, proc) in procs.iter_mut() {
        proc.stop().await;
    }
    procs.clear();
}

fn synth_bin_dir() -> Result<PathBuf, TunnelError> {
    let home = std::env::var("HOME").map_err(|_| TunnelError::config("HOME not set"))?;
    Ok(Path::new(&home).join(".synth").join("bin"))
}

pub fn get_cloudflared_path(prefer_system: bool) -> Option<PathBuf> {
    if let Ok(dir) = synth_bin_dir() {
        let candidate = dir.join("cloudflared");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    if prefer_system {
        if let Ok(path) = which::which("cloudflared") {
            return Some(path);
        }
    }
    let common = [
        PathBuf::from("/usr/local/bin/cloudflared"),
        PathBuf::from("/opt/homebrew/bin/cloudflared"),
        PathBuf::from(std::env::var("HOME").ok().unwrap_or_default()).join("bin/cloudflared"),
    ];
    for path in common {
        if path.exists() {
            return Some(path);
        }
    }
    None
}

pub async fn ensure_cloudflared_installed(force: bool) -> Result<PathBuf, TunnelError> {
    if !force {
        if let Some(path) = get_cloudflared_path(true) {
            return Ok(path);
        }
    }
    let dir = synth_bin_dir()?;
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| TunnelError::process(format!("mkdir failed: {e}")))?;
    let url = resolve_cloudflared_download_url().await?;
    let tmp = download_file(&url).await?;
    let target = dir.join("cloudflared");
    if tmp.extension().and_then(|s| s.to_str()) == Some("gz") {
        extract_gzip(&tmp, &target)?;
    } else if tmp.to_string_lossy().ends_with(".tar.gz") {
        extract_tarball(&tmp, &dir)?;
    } else {
        tokio::fs::copy(&tmp, &target)
            .await
            .map_err(|e| TunnelError::process(format!("copy failed: {e}")))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = tokio::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755)).await;
    }
    Ok(target)
}

pub async fn require_cloudflared() -> Result<PathBuf, TunnelError> {
    get_cloudflared_path(true).ok_or_else(|| TunnelError::config("cloudflared not found"))
}

async fn resolve_cloudflared_download_url() -> Result<String, TunnelError> {
    let system = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let platform = match system {
        "macos" | "darwin" => "macos",
        "linux" => "linux",
        "windows" => "windows",
        _ => {
            return Err(TunnelError::config(format!(
                "unsupported platform {system}"
            )))
        }
    };
    let arch_key = if arch == "aarch64" || arch == "arm64" {
        "arm64"
    } else {
        "amd64"
    };
    let url = format!("{CLOUDFLARED_RELEASES}/v1/{platform}/{arch_key}/versions/stable");
    let resp = reqwest::get(&url)
        .await
        .map_err(|e| TunnelError::process(format!("cloudflared metadata fetch failed: {e}")))?;
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| TunnelError::process(format!("cloudflared metadata parse failed: {e}")))?;
    json.get("url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| TunnelError::process("cloudflared metadata missing url"))
}

async fn download_file(url: &str) -> Result<PathBuf, TunnelError> {
    let resp = reqwest::get(url)
        .await
        .map_err(|e| TunnelError::process(format!("download failed: {e}")))?;
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| TunnelError::process(format!("download bytes failed: {e}")))?;
    let tmp = std::env::temp_dir().join(format!("cloudflared-{}.tmp", uuid::Uuid::new_v4()));
    tokio::fs::write(&tmp, bytes)
        .await
        .map_err(|e| TunnelError::process(format!("write failed: {e}")))?;
    Ok(tmp)
}

fn extract_gzip(src: &Path, target: &Path) -> Result<(), TunnelError> {
    let input = std::fs::File::open(src).map_err(|e| TunnelError::process(format!("{e}")))?;
    let mut gz = flate2::read::GzDecoder::new(input);
    let mut out =
        std::fs::File::create(target).map_err(|e| TunnelError::process(format!("{e}")))?;
    std::io::copy(&mut gz, &mut out).map_err(|e| TunnelError::process(format!("{e}")))?;
    Ok(())
}

fn extract_tarball(src: &Path, target_dir: &Path) -> Result<(), TunnelError> {
    let input = std::fs::File::open(src).map_err(|e| TunnelError::process(format!("{e}")))?;
    let gz = flate2::read::GzDecoder::new(input);
    let mut archive = tar::Archive::new(gz);
    archive
        .unpack(target_dir)
        .map_err(|e| TunnelError::process(format!("{e}")))?;
    Ok(())
}

async fn spawn_process(args: &[String]) -> Result<ManagedProcess, TunnelError> {
    let mut cmd = tokio::process::Command::new(&args[0]);
    cmd.args(&args[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| TunnelError::process(e.to_string()))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let logs = Arc::new(Mutex::new(VecDeque::with_capacity(200)));
    let mut stdout_task = None;
    let mut stderr_task = None;
    if let Some(out) = stdout {
        let logs = logs.clone();
        stdout_task = Some(tokio::spawn(async move {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&logs, &line);
            }
        }));
    }
    if let Some(err) = stderr {
        let logs = logs.clone();
        stderr_task = Some(tokio::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&logs, &line);
            }
        }));
    }
    Ok(ManagedProcess {
        child,
        logs,
        stdout_task,
        stderr_task,
    })
}

fn push_log(logs: &Arc<Mutex<VecDeque<String>>>, line: &str) {
    let mut guard = logs.lock();
    guard.push_back(line.to_string());
    if guard.len() > 200 {
        guard.pop_front();
    }
}

pub async fn open_quick_tunnel(
    port: u16,
    wait_s: f64,
) -> Result<(String, ManagedProcess), TunnelError> {
    let bin = require_cloudflared().await?;
    let args = vec![
        bin.to_string_lossy().to_string(),
        "tunnel".to_string(),
        "--config".to_string(),
        "/dev/null".to_string(),
        "--url".to_string(),
        format!("http://127.0.0.1:{port}"),
    ];
    let mut proc = spawn_process(&args).await?;
    let deadline = Instant::now() + Duration::from_secs_f64(wait_s);
    loop {
        if Instant::now() > deadline {
            let _ = proc.child.start_kill();
            return Err(TunnelError::process(
                "timed out waiting for quick tunnel URL",
            ));
        }
        if let Some(status) = proc.child.try_wait().ok().flatten() {
            return Err(TunnelError::process(format!(
                "cloudflared exited early with status {status}"
            )));
        }
        let url = {
            let logs = proc.logs.lock();
            logs.iter()
                .find_map(|line| URL_RE.find(line).map(|m| m.as_str().to_string()))
        };
        if let Some(url) = url {
            return Ok((url, proc));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

pub async fn open_quick_tunnel_with_dns_verification(
    port: u16,
    wait_s: f64,
    verify_dns: bool,
    api_key: Option<String>,
) -> Result<(String, ManagedProcess), TunnelError> {
    let (url, proc) = open_quick_tunnel(port, wait_s).await?;
    if verify_dns {
        verify_tunnel_dns_resolution(&url, "tunnel", 60.0, api_key).await?;
    }
    Ok((url, proc))
}

pub async fn open_managed_tunnel(tunnel_token: &str) -> Result<ManagedProcess, TunnelError> {
    let bin = require_cloudflared().await?;
    let args = vec![
        bin.to_string_lossy().to_string(),
        "tunnel".to_string(),
        "run".to_string(),
        "--token".to_string(),
        tunnel_token.to_string(),
    ];
    spawn_process(&args).await
}

pub async fn open_managed_tunnel_with_connection_wait(
    tunnel_token: &str,
    timeout_seconds: f64,
) -> Result<ManagedProcess, TunnelError> {
    let mut proc = open_managed_tunnel(tunnel_token).await?;
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds);
    let patterns = [
        Regex::new("Registered tunnel connection").unwrap(),
        Regex::new("Connection .* registered").unwrap(),
    ];
    loop {
        if Instant::now() > deadline {
            let _ = proc.child.start_kill();
            return Err(TunnelError::connector("cloudflared connection timeout"));
        }
        if let Some(status) = proc.child.try_wait().ok().flatten() {
            return Err(TunnelError::connector(format!(
                "cloudflared exited early with status {status}"
            )));
        }
        let connected = {
            let logs = proc.logs.lock();
            logs.iter()
                .any(|line| patterns.iter().any(|p| p.is_match(line)))
        };
        if connected {
            return Ok(proc);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunnelRotateResponse {
    pub tunnel_token: String,
    pub hostname: String,
    pub access_client_id: Option<String>,
    pub access_client_secret: Option<String>,
    pub dns_verified: Option<bool>,
}

pub async fn rotate_tunnel(
    api_key: &str,
    port: u16,
    backend_url: Option<String>,
) -> Result<TunnelRotateResponse, TunnelError> {
    let base = backend_url.unwrap_or_else(|| "https://api.usesynth.ai".to_string());
    let url = format!("{base}/api/v1/tunnels/rotate");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .pool_max_idle_per_host(20)
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .build()
        .map_err(|e| TunnelError::api(e.to_string()))?;
    let resp = client
        .post(url)
        .header("X-API-Key", api_key)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&serde_json::json!({
            "local_port": port,
            "local_host": "127.0.0.1",
        }))
        .send()
        .await
        .map_err(|e| TunnelError::api(e.to_string()))?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(TunnelError::api(format!("rotate failed: {}", text)));
    }
    let data: TunnelRotateResponse = resp
        .json()
        .await
        .map_err(|e| TunnelError::api(e.to_string()))?;
    Ok(data)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunnelCreateResponse {
    pub tunnel_token: String,
    pub hostname: String,
    pub access_client_id: Option<String>,
    pub access_client_secret: Option<String>,
    pub dns_verified: Option<bool>,
}

pub async fn create_tunnel(
    api_key: &str,
    port: u16,
    subdomain: Option<String>,
) -> Result<TunnelCreateResponse, TunnelError> {
    let url = "https://api.usesynth.ai/api/v1/tunnels/";
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .pool_max_idle_per_host(20)
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .build()
        .map_err(|e| TunnelError::api(e.to_string()))?;
    let resp = client
        .post(url)
        .header("X-API-Key", api_key)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&serde_json::json!({
            "subdomain": subdomain.unwrap_or_else(|| format!("tunnel-{port}")),
            "local_port": port,
            "local_host": "127.0.0.1",
        }))
        .send()
        .await
        .map_err(|e| TunnelError::api(e.to_string()))?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(TunnelError::api(format!("create failed: {}", text)));
    }
    let data: TunnelCreateResponse = resp
        .json()
        .await
        .map_err(|e| TunnelError::api(e.to_string()))?;
    Ok(data)
}

pub async fn wait_for_health_check(
    host: &str,
    port: u16,
    api_key: Option<String>,
    timeout: f64,
) -> Result<(), TunnelError> {
    let url = format!("http://{host}:{port}/health");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(10)
        .connect_timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .map_err(|e| TunnelError::local(e.to_string()))?;
    let start = Instant::now();
    let headers = api_key.map(|k| ("X-API-Key", k));
    while start.elapsed() < Duration::from_secs_f64(timeout) {
        let mut req = client.get(&url);
        if let Some((k, v)) = headers.clone() {
            req = req.header(k, v);
        }
        if let Ok(resp) = req.send().await {
            let status = resp.status().as_u16();
            if status == 200 || status == 400 {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(TunnelError::local(format!(
        "health check failed: {url} not ready after {timeout}s"
    )))
}

pub async fn resolve_hostname_with_explicit_resolvers(
    hostname: &str,
) -> Result<std::net::IpAddr, TunnelError> {
    use trust_dns_resolver::config::{NameServerConfig, Protocol, ResolverConfig, ResolverOpts};
    use trust_dns_resolver::TokioAsyncResolver;

    let servers = vec![("1.1.1.1:53", "1.1.1.1"), ("8.8.8.8:53", "8.8.8.8")];
    for (socket, _) in servers {
        if let Ok(addr) = socket.parse() {
            let config = ResolverConfig::from_parts(
                None,
                vec![],
                vec![NameServerConfig {
                    socket_addr: addr,
                    protocol: Protocol::Udp,
                    tls_dns_name: None,
                    trust_negative_responses: false,
                    bind_addr: None,
                }],
            );
            let resolver = TokioAsyncResolver::tokio(config, ResolverOpts::default());
            if let Ok(lookup) = resolver.lookup_ip(hostname).await {
                if let Some(ip) = lookup.iter().next() {
                    return Ok(ip);
                }
            }
        }
    }
    // Fallback to system resolver
    let resolver = TokioAsyncResolver::tokio(ResolverConfig::default(), ResolverOpts::default());
    let lookup = resolver
        .lookup_ip(hostname)
        .await
        .map_err(|e| TunnelError::dns(e.to_string()))?;
    lookup
        .iter()
        .next()
        .ok_or_else(|| TunnelError::dns("no ip resolved"))
}

pub async fn verify_tunnel_dns_resolution(
    tunnel_url: &str,
    _name: &str,
    timeout_seconds: f64,
    api_key: Option<String>,
) -> Result<(), TunnelError> {
    let parsed =
        url::Url::parse(tunnel_url).map_err(|e| TunnelError::dns(format!("invalid url: {e}")))?;
    let hostname = parsed
        .host_str()
        .ok_or_else(|| TunnelError::dns("missing hostname"))?;
    if hostname == "localhost" || hostname == "127.0.0.1" {
        return Ok(());
    }
    let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds);
    let mut last_err: Option<String> = None;
    loop {
        if Instant::now() > deadline {
            return Err(TunnelError::dns(format!(
                "dns verification timeout: {} ({:?})",
                hostname, last_err
            )));
        }
        let ip = resolve_hostname_with_explicit_resolvers(hostname).await?;
        let port = if parsed.scheme() == "http" { 80 } else { 443 };
        let builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .pool_max_idle_per_host(10)
            .connect_timeout(Duration::from_secs(5))
            .danger_accept_invalid_certs(true)
            .resolve(hostname, (ip, port).into());
        let client = builder
            .build()
            .map_err(|e| TunnelError::dns(e.to_string()))?;
        let mut req = client.get(parsed.clone());
        if let Some(key) = api_key.clone() {
            req = req.header("X-API-Key", key);
        }
        match req.send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                if matches!(status, 200 | 400 | 401 | 403 | 404 | 405 | 502) {
                    return Ok(());
                }
                last_err = Some(format!("status {status}"));
            }
            Err(e) => {
                last_err = Some(e.to_string());
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

pub async fn stop_tunnel(mut proc: ManagedProcess) {
    let _ = proc.child.start_kill();
    let _ = proc.child.wait().await;
}
