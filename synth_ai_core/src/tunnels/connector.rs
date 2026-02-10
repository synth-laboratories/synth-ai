use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;
use tokio::task::JoinHandle;

use crate::tunnels::cloudflared::require_cloudflared;
use crate::tunnels::errors::TunnelError;
use crate::tunnels::types::{ConnectorState, ConnectorStatus};

static CONNECT_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new("Registered tunnel connection").unwrap(),
        Regex::new("Connection .* registered").unwrap(),
    ]
});

static ERROR_PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    vec![
        (
            Regex::new("failed to connect").unwrap(),
            "Connection failed",
        ),
        (Regex::new("invalid.*token").unwrap(), "Invalid token"),
        (
            Regex::new("tunnel.*not.*found").unwrap(),
            "Tunnel not found",
        ),
        (Regex::new("unauthorized").unwrap(), "Unauthorized"),
        (Regex::new("rate.*limit").unwrap(), "Rate limited"),
    ]
});

pub struct TunnelConnector {
    process: Option<Child>,
    state: ConnectorState,
    connected_at: Option<DateTime<Utc>>,
    error: Option<String>,
    current_token: Option<String>,
    logs: Arc<Mutex<VecDeque<String>>>,
    stdout_task: Option<JoinHandle<()>>,
    stderr_task: Option<JoinHandle<()>>,
    idle_timeout: Duration,
    active_leases: HashSet<String>,
    idle_task: Option<JoinHandle<()>>,
}

impl TunnelConnector {
    pub fn new(idle_timeout: Duration) -> Self {
        Self {
            process: None,
            state: ConnectorState::Stopped,
            connected_at: None,
            error: None,
            current_token: None,
            logs: Arc::new(Mutex::new(VecDeque::with_capacity(200))),
            stdout_task: None,
            stderr_task: None,
            idle_timeout,
            active_leases: HashSet::new(),
            idle_task: None,
        }
    }

    pub fn status(&self) -> ConnectorStatus {
        ConnectorStatus {
            state: self.state.clone(),
            pid: self.process.as_ref().and_then(|c| c.id()),
            connected_at: self.connected_at,
            error: self.error.clone(),
        }
    }

    pub fn is_connected(&self) -> bool {
        self.state == ConnectorState::Connected
    }

    pub async fn start(
        &mut self,
        token: &str,
        timeout: Duration,
        force_restart: bool,
    ) -> Result<(), TunnelError> {
        if self.state != ConnectorState::Stopped
            && self.current_token.as_deref() == Some(token)
            && !force_restart
        {
            self.cancel_idle_timer();
            return Ok(());
        }
        if self.state != ConnectorState::Stopped {
            self.stop().await?;
        }
        let bin = require_cloudflared().await?;
        let mut cmd = tokio::process::Command::new(bin);
        cmd.arg("tunnel").arg("run").arg("--token").arg(token);
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| TunnelError::connector(e.to_string()))?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let logs = self.logs.clone();
        if let Some(out) = stdout {
            let logs = logs.clone();
            self.stdout_task = Some(tokio::spawn(async move {
                let mut lines = BufReader::new(out).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    push_log(&logs, &line);
                }
            }));
        }
        if let Some(err) = stderr {
            let logs = logs.clone();
            self.stderr_task = Some(tokio::spawn(async move {
                let mut lines = BufReader::new(err).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    push_log(&logs, &line);
                }
            }));
        }

        self.state = ConnectorState::Starting;
        self.error = None;
        self.current_token = Some(token.to_string());
        self.connected_at = None;
        self.process = Some(child);

        self.wait_for_connection(timeout).await?;
        self.state = ConnectorState::Connected;
        self.connected_at = Some(Utc::now());
        Ok(())
    }

    async fn wait_for_connection(&mut self, timeout: Duration) -> Result<(), TunnelError> {
        let deadline = Instant::now() + timeout;
        loop {
            if Instant::now() > deadline {
                return Err(TunnelError::connector(
                    "timeout waiting for cloudflared connection",
                ));
            }
            if let Some(proc) = &mut self.process {
                if let Ok(Some(status)) = proc.try_wait() {
                    return Err(TunnelError::connector(format!(
                        "cloudflared exited early: {status}"
                    )));
                }
            }
            {
                let logs = self.logs.lock();
                for line in logs.iter() {
                    if CONNECT_PATTERNS.iter().any(|p| p.is_match(line)) {
                        return Ok(());
                    }
                    for (p, msg) in ERROR_PATTERNS.iter() {
                        if p.is_match(line) {
                            return Err(TunnelError::connector(msg.to_string()));
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn stop(&mut self) -> Result<(), TunnelError> {
        let mut proc_opt = self.prepare_stop();
        if let Some(proc) = &mut proc_opt {
            let _ = proc.start_kill();
            let _ = proc.wait().await;
        }
        Ok(())
    }

    pub fn register_lease(&mut self, lease_id: &str) {
        self.active_leases.insert(lease_id.to_string());
        self.cancel_idle_timer();
    }

    pub fn unregister_lease(&mut self, lease_id: &str) {
        self.active_leases.remove(lease_id);
        if self.active_leases.is_empty() {
            self.start_idle_timer();
        }
    }

    pub fn get_logs(&self, lines: usize) -> Vec<String> {
        let logs = self.logs.lock();
        logs.iter().rev().take(lines).cloned().collect()
    }

    fn cancel_idle_timer(&mut self) {
        if let Some(task) = self.idle_task.take() {
            task.abort();
        }
    }

    fn start_idle_timer(&mut self) {
        let timeout = self.idle_timeout;
        self.idle_task = Some(tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            let connector = get_connector();
            let mut proc_opt = None;
            {
                let mut guard = connector.lock();
                if guard.active_leases.is_empty() {
                    proc_opt = guard.prepare_stop();
                }
            }
            if let Some(mut proc) = proc_opt {
                let _ = proc.start_kill();
                let _ = proc.wait().await;
            }
        }));
    }

    fn prepare_stop(&mut self) -> Option<Child> {
        self.cancel_idle_timer();
        let proc = self.process.take();
        self.state = ConnectorState::Stopped;
        self.current_token = None;
        self.active_leases.clear();
        proc
    }
}

fn push_log(logs: &Arc<Mutex<VecDeque<String>>>, line: &str) {
    let mut guard = logs.lock();
    guard.push_back(line.to_string());
    if guard.len() > 200 {
        guard.pop_front();
    }
}

static CONNECTOR: Lazy<Mutex<TunnelConnector>> =
    Lazy::new(|| Mutex::new(TunnelConnector::new(Duration::from_secs(120))));

pub fn get_connector() -> &'static Mutex<TunnelConnector> {
    &CONNECTOR
}
