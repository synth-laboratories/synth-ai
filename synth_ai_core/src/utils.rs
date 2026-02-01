use crate::errors::{CoreError, CoreResult};
use serde_json::Value;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const PRIVATE_DIR_MODE: u32 = 0o700;
pub const PRIVATE_FILE_MODE: u32 = 0o600;

pub fn strip_json_comments(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    let mut escape = false;
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = if i + 1 < chars.len() {
            chars[i + 1]
        } else {
            '\0'
        };

        if in_line_comment {
            if c == '\n' {
                in_line_comment = false;
                result.push(c);
            }
            i += 1;
            continue;
        }

        if in_block_comment {
            if c == '*' && next == '/' {
                in_block_comment = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }

        if in_string {
            result.push(c);
            if c == '"' && !escape {
                in_string = false;
            }
            escape = c == '\\' && !escape;
            i += 1;
            continue;
        }

        if c == '/' && next == '/' {
            in_line_comment = true;
            i += 2;
            continue;
        }

        if c == '/' && next == '*' {
            in_block_comment = true;
            i += 2;
            continue;
        }

        if c == '"' {
            in_string = true;
            escape = false;
        }

        result.push(c);
        i += 1;
    }

    result
}

pub fn create_and_write_json(path: &Path, content: &Value) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        + "\n";
    fs::write(path, payload)
}

pub fn load_json_to_value(path: &Path) -> Value {
    if !path.exists() {
        return Value::Object(Default::default());
    }
    let raw = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(_) => return Value::Object(Default::default()),
    };
    let stripped = strip_json_comments(&raw);
    serde_json::from_str(&stripped).unwrap_or_else(|_| Value::Object(Default::default()))
}

pub fn repo_root() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
}

pub fn synth_home_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".synth-ai")
}

pub fn synth_user_config_path() -> PathBuf {
    synth_home_dir().join("user_config.json")
}

pub fn synth_localapi_config_path() -> PathBuf {
    synth_home_dir().join("localapi_config.json")
}

pub fn synth_bin_dir() -> PathBuf {
    synth_home_dir().join("bin")
}

pub fn is_file_type(path: &Path, ext: &str) -> bool {
    let mut ext = ext.to_string();
    if !ext.starts_with('.') {
        ext.insert(0, '.');
    }
    path.is_file()
        && path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            == Some(ext)
}

pub fn validate_file_type(path: &Path, ext: &str) -> CoreResult<()> {
    if !is_file_type(path, ext) {
        return Err(CoreError::InvalidInput(format!(
            "{} is not a {} file",
            path.display(),
            ext
        )));
    }
    Ok(())
}

pub fn is_hidden_path(path: &Path, root: &Path) -> bool {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.components()
        .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
}

pub fn get_bin_path(name: &str) -> Option<PathBuf> {
    let path_var = env::var("PATH").ok()?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let exe = candidate.with_extension("exe");
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
}

pub fn get_home_config_file_paths(dir_name: &str, file_extension: &str) -> Vec<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = home.join(dir_name);
    let mut results = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == file_extension {
                        results.push(path);
                    }
                }
            }
        }
    }
    results
}

pub fn find_config_path(bin: &Path, home_subdir: &str, filename: &str) -> Option<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let home_candidate = home.join(home_subdir).join(filename);
    if home_candidate.exists() {
        return Some(home_candidate);
    }
    let local_candidate = bin
        .parent()
        .unwrap_or(Path::new("."))
        .join(home_subdir)
        .join(filename);
    if local_candidate.exists() {
        return Some(local_candidate);
    }
    None
}

pub fn compute_import_paths(app: &Path, repo_root: Option<&Path>) -> Vec<String> {
    let app_dir = app.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut initial_dirs: Vec<PathBuf> = vec![app_dir.clone()];
    if app_dir.join("__init__.py").exists() {
        if let Some(parent) = app_dir.parent() {
            initial_dirs.push(parent.to_path_buf());
        }
    }
    if let Some(root) = repo_root {
        initial_dirs.push(root.to_path_buf());
    }

    let mut unique_dirs: Vec<String> = Vec::new();
    for dir in initial_dirs {
        let dir_str = dir.to_string_lossy().to_string();
        if !dir_str.is_empty() && !unique_dirs.contains(&dir_str) {
            unique_dirs.push(dir_str);
        }
    }

    if let Ok(existing) = env::var("PYTHONPATH") {
        for segment in env::split_paths(&existing) {
            let segment_str = segment.to_string_lossy().to_string();
            if !segment_str.is_empty() && !unique_dirs.contains(&segment_str) {
                unique_dirs.push(segment_str);
            }
        }
    }

    unique_dirs
}

pub fn cleanup_paths(file: &Path, dir: &Path) -> CoreResult<()> {
    if !file.starts_with(dir) {
        return Err(CoreError::InvalidInput(format!(
            "{} is not inside {}",
            file.display(),
            dir.display()
        )));
    }
    let _ = fs::remove_file(file);
    let _ = fs::remove_dir_all(dir);
    Ok(())
}

fn set_permissions(path: &Path, mode: u32) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(mode);
        fs::set_permissions(path, perms)?;
    }
    #[cfg(not(unix))]
    {
        let _ = mode;
    }
    Ok(())
}

pub fn ensure_private_dir(path: &Path) -> io::Result<()> {
    fs::create_dir_all(path)?;
    let _ = set_permissions(path, PRIVATE_DIR_MODE);
    Ok(())
}

pub fn write_private_text(path: &Path, content: &str, mode: u32) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        ensure_private_dir(parent)?;
        let mut tmp = tempfile::Builder::new()
            .prefix(&format!(
                "{}.",
                path.file_name().unwrap_or_default().to_string_lossy()
            ))
            .tempfile_in(parent)?;
        let _ = set_permissions(tmp.path(), mode);
        tmp.write_all(content.as_bytes())?;
        tmp.flush()?;
        let _ = tmp.as_file().sync_all();
        tmp.persist(path).map_err(|e| e.error)?;
        let _ = set_permissions(path, mode);
    }
    Ok(())
}

pub fn write_private_json(path: &Path, data: &Value) -> io::Result<()> {
    let payload = serde_json::to_string_pretty(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        + "\n";
    write_private_text(path, &payload, PRIVATE_FILE_MODE)
}

pub fn should_filter_log_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lowered = trimmed.to_lowercase();
    let substrings = [
        "codex_otel::otel_event_manager",
        "event.kind=response.reasoning_summary_text.delta",
        "event.name=\"codex.sse_event\"",
        "codex_otel",
    ];
    for substr in substrings {
        if lowered.contains(substr) {
            return true;
        }
    }
    let patterns = [
        r"(?i).*codex_otel::otel_event_manager.*",
        r"(?i).*event\.kind=response\.reasoning_summary_text\.delta.*",
        r#"(?i).*event\.name="codex\.sse_event".*"#,
        r"(?i)^\d{4}-\d{2}-\d{2}t.*codex_otel.*",
    ];
    for pat in patterns {
        if let Ok(re) = regex::Regex::new(pat) {
            if re.is_match(trimmed) {
                return true;
            }
        }
    }
    false
}
