use std::net::TcpListener;
use std::process::Command;
use std::thread;
use std::time::Duration;

use crate::tunnels::errors::TunnelError;

#[derive(Debug, Clone, Copy)]
pub enum PortConflictBehavior {
    Fail,
    Evict,
    FindNew,
}

pub fn is_port_available(port: u16, host: &str) -> bool {
    let hosts = if host == "0.0.0.0" {
        vec!["0.0.0.0", "127.0.0.1"]
    } else {
        vec![host]
    };
    for h in hosts {
        let addr = format!("{h}:{port}");
        if TcpListener::bind(addr).is_err() {
            return false;
        }
    }
    true
}

pub fn find_available_port(
    start_port: u16,
    host: &str,
    max_attempts: u16,
) -> Result<u16, TunnelError> {
    for offset in 0..max_attempts {
        let port = start_port.saturating_add(offset);
        if is_port_available(port, host) {
            return Ok(port);
        }
    }
    Err(TunnelError::config(format!(
        "no available port found starting from {start_port}"
    )))
}

pub fn kill_port(port: u16) -> Result<bool, TunnelError> {
    if is_port_available(port, "0.0.0.0") {
        return Ok(false);
    }
    if cfg!(target_os = "windows") {
        let output = Command::new("netstat")
            .arg("-ano")
            .output()
            .map_err(|e| TunnelError::process(format!("netstat failed: {e}")))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains(&format!(":{port}")) && line.contains("LISTENING") {
                if let Some(pid) = line.split_whitespace().last() {
                    let _ = Command::new("taskkill").args(["/F", "/PID", pid]).output();
                    return Ok(true);
                }
            }
        }
        return Ok(false);
    }

    let output = Command::new("lsof")
        .args(["-ti", &format!(":{port}")])
        .output()
        .map_err(|e| TunnelError::process(format!("lsof failed: {e}")))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Ok(false);
    }
    for pid in stdout.split_whitespace() {
        let _ = Command::new("kill").args(["-9", pid]).output();
    }
    Ok(true)
}

pub fn acquire_port(
    port: u16,
    on_conflict: PortConflictBehavior,
    host: &str,
    max_search: u16,
) -> Result<u16, TunnelError> {
    if is_port_available(port, host) {
        return Ok(port);
    }
    match on_conflict {
        PortConflictBehavior::Fail => Err(TunnelError::config(format!(
            "port {port} is already in use"
        ))),
        PortConflictBehavior::Evict => {
            let killed = kill_port(port)?;
            if killed {
                thread::sleep(Duration::from_millis(500));
                if is_port_available(port, host) {
                    return Ok(port);
                }
            }
            Err(TunnelError::process(format!(
                "failed to evict process on port {port}"
            )))
        }
        PortConflictBehavior::FindNew => find_available_port(port, host, max_search),
    }
}
