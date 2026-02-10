use chrono::{DateTime, Local};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TRACE_DB_DIR: &str = "traces";
const TRACE_DB_BASENAME: &str = "turso_task_app_traces";

fn canonical_trace_db_name(timestamp: Option<DateTime<Local>>) -> String {
    match timestamp {
        Some(ts) => format!(
            "{}_{}.db",
            TRACE_DB_BASENAME,
            ts.format("%Y-%m-%d_%H-%M-%S")
        ),
        None => format!("{}.db", TRACE_DB_BASENAME),
    }
}

pub fn tracing_env_enabled(default: bool) -> bool {
    let raw = env::var("TASKAPP_TRACING_ENABLED").ok();
    let raw = match raw {
        Some(value) => value.to_lowercase(),
        None => return default,
    };

    match raw.as_str() {
        "1" | "true" | "t" | "yes" | "y" | "on" => true,
        "0" | "false" | "f" | "no" | "n" | "off" => false,
        _ => default,
    }
}

pub fn resolve_tracing_db_url() -> Option<String> {
    if let Ok(url) = env::var("TURSO_LOCAL_DB_URL") {
        if !url.trim().is_empty() {
            return Some(url);
        }
    }
    if let Ok(url) = env::var("LIBSQL_URL") {
        if !url.trim().is_empty() {
            return Some(url);
        }
    }
    if let Ok(url) = env::var("SYNTH_TRACES_DB") {
        if !url.trim().is_empty() {
            return Some(url);
        }
    }

    let base_dir = PathBuf::from(TRACE_DB_DIR);
    let _ = fs::create_dir_all(&base_dir);
    let candidate = base_dir.join(canonical_trace_db_name(Some(Local::now())));
    if let Some(path_str) = candidate.to_str() {
        env::set_var("TASKAPP_TRACE_DB_PATH", path_str);
        env::set_var("SQLD_DB_PATH", path_str);
    }

    let default_url =
        env::var("LIBSQL_DEFAULT_URL").unwrap_or_else(|_| "http://127.0.0.1:8081".to_string());
    Some(default_url)
}

pub fn resolve_sft_output_dir() -> Option<String> {
    let raw = env::var("TASKAPP_SFT_OUTPUT_DIR")
        .or_else(|_| env::var("SFT_OUTPUT_DIR"))
        .ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let path = Path::new(raw.trim());
    if fs::create_dir_all(path).is_err() {
        return None;
    }
    Some(path.to_string_lossy().to_string())
}

pub fn unique_sft_path(base_dir: &str, run_id: &str) -> Option<String> {
    let base = Path::new(base_dir);
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let name = format!("{}_{}.jsonl", run_id, timestamp);
    let path = base.join(name);
    path.to_str().map(|s| s.to_string())
}
