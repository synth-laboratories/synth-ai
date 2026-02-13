use crate::data::{
    ApplicationErrorType, ApplicationStatus, ContextOverride, ContextOverrideStatus,
};
use crate::errors::CoreError;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const MAX_FILE_SIZE_BYTES: usize = 1 * 1024 * 1024;
pub const MAX_TOTAL_SIZE_BYTES: usize = 10 * 1024 * 1024;
pub const MAX_FILES_PER_OVERRIDE: usize = 20;
pub const MAX_ENV_VARS: usize = 50;
pub const MAX_ENV_VAR_VALUE_LENGTH: usize = 10 * 1024;
pub const PREFLIGHT_SCRIPT_TIMEOUT_SECONDS: u64 = 60;

const CODEX_SKILLS_WORKSPACE_PATH: &str = ".codex/skills.yaml";
const CODEX_SKILLS_GLOBAL_PATH: &str = "~/.codex/skills.yaml";
const OPENCODE_SKILLS_WORKSPACE_PATH: &str = ".opencode/skills.yaml";
const OPENCODE_SKILLS_GLOBAL_PATH: &str = "~/.opencode/skills.yaml";

pub fn get_agent_skills_path(agent: &str, global: bool) -> String {
    let agent_lower = agent.trim().to_lowercase();
    match agent_lower.as_str() {
        "codex" => {
            if global {
                CODEX_SKILLS_GLOBAL_PATH.to_string()
            } else {
                CODEX_SKILLS_WORKSPACE_PATH.to_string()
            }
        }
        "opencode" => {
            if global {
                OPENCODE_SKILLS_GLOBAL_PATH.to_string()
            } else {
                OPENCODE_SKILLS_WORKSPACE_PATH.to_string()
            }
        }
        other => {
            if global {
                format!("~/.{other}/skills.yaml")
            } else {
                format!(".{other}/skills.yaml")
            }
        }
    }
}

fn expand_tilde(path: &str) -> Option<PathBuf> {
    if !path.starts_with('~') {
        return None;
    }
    let home = env::var("HOME").ok()?;
    let rest = path.trim_start_matches('~');
    Some(PathBuf::from(home).join(rest.trim_start_matches('/')))
}

fn is_path_safe(
    path: &str,
    workspace_dir: &Path,
    allow_global: bool,
) -> Result<PathBuf, (ApplicationErrorType, String)> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err((ApplicationErrorType::Validation, "Empty path".to_string()));
    }
    if trimmed.contains("..") {
        return Err((
            ApplicationErrorType::PathTraversal,
            format!("Path traversal not allowed: {trimmed}"),
        ));
    }

    let workspace_prefix = workspace_dir.to_string_lossy();
    if trimmed.starts_with('/') && !trimmed.starts_with(workspace_prefix.as_ref()) {
        if allow_global {
            if let Ok(home) = env::var("HOME") {
                if trimmed.starts_with(&home) {
                    return Ok(PathBuf::from(trimmed));
                }
            }
        }
        return Err((
            ApplicationErrorType::PathTraversal,
            format!("Absolute paths outside workspace not allowed: {trimmed}"),
        ));
    }

    if trimmed.starts_with('~') {
        if !allow_global {
            return Err((
                ApplicationErrorType::PathTraversal,
                format!("Global paths require explicit opt-in: {trimmed}"),
            ));
        }
        if let Some(expanded) = expand_tilde(trimmed) {
            return Ok(expanded);
        }
    }

    let final_path = if trimmed.starts_with('/') {
        PathBuf::from(trimmed)
    } else {
        workspace_dir.join(trimmed)
    };

    Ok(final_path)
}

fn apply_file_artifact(
    path: &str,
    content: &str,
    workspace_dir: &Path,
    allow_global: bool,
) -> Result<usize, (ApplicationErrorType, String)> {
    let final_path = is_path_safe(path, workspace_dir, allow_global)?;

    let content_bytes = content.as_bytes();
    if content_bytes.len() > MAX_FILE_SIZE_BYTES {
        return Err((
            ApplicationErrorType::SizeLimit,
            format!(
                "File too large: {} > {} bytes",
                content_bytes.len(),
                MAX_FILE_SIZE_BYTES
            ),
        ));
    }

    if let Some(parent) = final_path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            return Err((
                ApplicationErrorType::Permission,
                format!("Failed to create directory: {err}"),
            ));
        }
    }

    if let Err(err) = fs::write(&final_path, content_bytes) {
        return Err((
            ApplicationErrorType::Permission,
            format!("Failed to write file: {err}"),
        ));
    }

    Ok(content_bytes.len())
}

fn validate_env_vars(
    env_vars: &HashMap<String, String>,
) -> (Vec<String>, Vec<(ApplicationErrorType, String)>) {
    let mut applied = Vec::new();
    let mut errors = Vec::new();

    if env_vars.len() > MAX_ENV_VARS {
        errors.push((
            ApplicationErrorType::SizeLimit,
            format!("Too many env vars: {} > {}", env_vars.len(), MAX_ENV_VARS),
        ));
    }

    for (key, value) in env_vars {
        let first_char = key.chars().next();
        if first_char.is_none()
            || !(first_char.unwrap().is_ascii_alphabetic() || first_char.unwrap() == '_')
        {
            errors.push((
                ApplicationErrorType::Validation,
                format!("Invalid env var name (must start with letter or underscore): {key}"),
            ));
            continue;
        }

        if value.len() > MAX_ENV_VAR_VALUE_LENGTH {
            errors.push((
                ApplicationErrorType::SizeLimit,
                format!(
                    "Value too large for {key}: {} > {}",
                    value.len(),
                    MAX_ENV_VAR_VALUE_LENGTH
                ),
            ));
            continue;
        }

        applied.push(key.clone());
    }

    (applied, errors)
}

fn run_preflight_script(
    script_content: &str,
    workspace_dir: &Path,
    timeout_seconds: u64,
    env_vars: &HashMap<String, String>,
) -> (
    Option<bool>,
    Option<i64>,
    Option<ApplicationErrorType>,
    Option<String>,
) {
    if !script_content.trim_start().starts_with("#!") {
        return (
            Some(false),
            None,
            Some(ApplicationErrorType::Validation),
            Some("Preflight script must start with a shebang (e.g., #!/bin/bash)".to_string()),
        );
    }

    if script_content.as_bytes().len() > MAX_FILE_SIZE_BYTES {
        return (
            Some(false),
            None,
            Some(ApplicationErrorType::SizeLimit),
            Some(format!(
                "Script too large: {} bytes",
                script_content.as_bytes().len()
            )),
        );
    }

    let script_path = workspace_dir.join(".synth_preflight.sh");
    if let Err(err) = fs::write(&script_path, script_content) {
        return (
            Some(false),
            None,
            Some(ApplicationErrorType::Permission),
            Some(format!("Failed to write script: {err}")),
        );
    }
    let _ = fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755));

    let mut cmd = Command::new("bash");
    cmd.arg(&script_path)
        .current_dir(workspace_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    for (k, v) in env_vars {
        cmd.env(k, v);
    }

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            let _ = fs::remove_file(&script_path);
            return (
                Some(false),
                None,
                Some(ApplicationErrorType::Runtime),
                Some(format!("Script execution failed: {err}")),
            );
        }
    };

    let start = Instant::now();
    let timeout = Duration::from_secs(timeout_seconds);
    let mut timed_out = false;
    let mut exit_code = None;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                exit_code = status.code();
                break;
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => {
                break;
            }
        }
    }

    let duration_ms = start.elapsed().as_millis() as i64;
    let _ = fs::remove_file(&script_path);

    if timed_out {
        return (
            Some(false),
            Some(duration_ms),
            Some(ApplicationErrorType::Timeout),
            Some(format!("Script timed out after {}s", timeout_seconds)),
        );
    }

    if let Some(code) = exit_code {
        if code == 0 {
            return (Some(true), Some(duration_ms), None, None);
        }
        return (
            Some(false),
            Some(duration_ms),
            Some(ApplicationErrorType::Runtime),
            Some(format!("Script exited with code {code}")),
        );
    }

    (
        Some(false),
        Some(duration_ms),
        Some(ApplicationErrorType::Runtime),
        Some("Script execution failed".to_string()),
    )
}

pub fn apply_context_overrides(
    overrides: &[ContextOverride],
    workspace_dir: &Path,
    allow_global: bool,
    override_bundle_id: Option<&str>,
) -> Result<Vec<ContextOverrideStatus>, CoreError> {
    if overrides.is_empty() {
        return Ok(Vec::new());
    }

    fs::create_dir_all(workspace_dir).map_err(|e| CoreError::InvalidInput(e.to_string()))?;

    let mut results = Vec::with_capacity(overrides.len());

    for (idx, override_item) in overrides.iter().enumerate() {
        let start = Instant::now();
        let override_id = override_item
            .override_id
            .clone()
            .or_else(|| override_bundle_id.map(|id| format!("{id}_{idx}")));

        let mut status = ContextOverrideStatus::success(override_id.clone());
        let mut has_success = false;
        let mut has_failure = false;
        let mut first_error: Option<(ApplicationErrorType, String)> = None;

        if !override_item.file_artifacts.is_empty() {
            if override_item.file_artifacts.len() > MAX_FILES_PER_OVERRIDE {
                has_failure = true;
                first_error.get_or_insert((
                    ApplicationErrorType::SizeLimit,
                    format!(
                        "Too many files: {} > {}",
                        override_item.file_artifacts.len(),
                        MAX_FILES_PER_OVERRIDE
                    ),
                ));
            }

            for (path, content) in override_item.file_artifacts.iter() {
                match apply_file_artifact(path, content, workspace_dir, allow_global) {
                    Ok(_) => {
                        has_success = true;
                        status.files_applied.push(path.clone());
                    }
                    Err((err_type, message)) => {
                        has_failure = true;
                        status.files_failed.push(path.clone());
                        first_error.get_or_insert((err_type, message));
                    }
                }
            }
        }

        if let Some(script) = override_item.preflight_script.as_ref() {
            let (success_opt, duration_ms, err_type, err_message) = run_preflight_script(
                script,
                workspace_dir,
                PREFLIGHT_SCRIPT_TIMEOUT_SECONDS,
                &override_item.env_vars,
            );
            status.preflight_succeeded = success_opt;
            if let Some(ms) = duration_ms {
                status.duration_ms = Some(ms);
            }
            if success_opt == Some(true) {
                has_success = true;
            } else {
                has_failure = true;
                if let Some(err_type) = err_type {
                    if let Some(message) = err_message {
                        first_error.get_or_insert((err_type, message));
                    }
                }
            }
        }

        if !override_item.env_vars.is_empty() {
            let (applied, errors) = validate_env_vars(&override_item.env_vars);
            if !applied.is_empty() {
                has_success = true;
                status.env_vars_applied.extend(applied);
            }
            if !errors.is_empty() {
                has_failure = true;
                let (err_type, message) = errors[0].clone();
                first_error.get_or_insert((err_type, message));
            }
        }

        status.overall_status = if has_failure {
            if has_success {
                ApplicationStatus::Partial
            } else {
                ApplicationStatus::Failed
            }
        } else {
            ApplicationStatus::Applied
        };

        if let Some((err_type, message)) = first_error {
            status.error_type = Some(err_type);
            status.error_message = Some(message);
        }

        if status.duration_ms.is_none() {
            status.duration_ms = Some(start.elapsed().as_millis() as i64);
        }

        results.push(status);
    }

    Ok(results)
}

pub fn get_applied_env_vars(overrides: &[ContextOverride]) -> HashMap<String, String> {
    let mut merged = HashMap::new();
    for override_item in overrides {
        for (k, v) in override_item.env_vars.iter() {
            merged.insert(k.clone(), v.clone());
        }
    }
    merged
}
