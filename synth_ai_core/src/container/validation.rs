use crate::data::{Artifact, ContextOverride};
use crate::errors::CoreError;
use serde_json::Value;

pub const MAX_INLINE_ARTIFACT_BYTES: usize = 64 * 1024;
pub const MAX_TOTAL_INLINE_ARTIFACTS_BYTES: usize = 256 * 1024;
pub const MAX_ARTIFACTS_PER_ROLLOUT: usize = 10;
pub const MAX_ARTIFACT_METADATA_BYTES: usize = 16 * 1024;
pub const MAX_ARTIFACT_CONTENT_TYPE_LENGTH: usize = 128;
pub const MAX_CONTEXT_SNAPSHOT_BYTES: usize = 1 * 1024 * 1024;
pub const MAX_CONTEXT_OVERRIDES_PER_ROLLOUT: usize = 20;

pub fn validate_artifact_size(artifact: &Artifact, max_bytes: usize) -> Result<(), CoreError> {
    artifact
        .validate_size(max_bytes as i64)
        .map_err(CoreError::InvalidInput)
}

pub fn validate_artifacts_list(artifacts: &[Artifact]) -> Result<(), CoreError> {
    if artifacts.len() > MAX_ARTIFACTS_PER_ROLLOUT {
        return Err(CoreError::InvalidInput(format!(
            "Too many artifacts: {} > {}",
            artifacts.len(),
            MAX_ARTIFACTS_PER_ROLLOUT
        )));
    }

    let mut total_size = 0usize;

    for artifact in artifacts {
        let size = match &artifact.content {
            crate::data::ArtifactContent::Text(text) => text.as_bytes().len(),
            crate::data::ArtifactContent::Structured(map) => {
                serde_json::to_string(map).map(|s| s.len()).unwrap_or(0)
            }
        };
        total_size += size;

        if let Some(content_type) = &artifact.content_type {
            if content_type.len() > MAX_ARTIFACT_CONTENT_TYPE_LENGTH {
                return Err(CoreError::InvalidInput(format!(
                    "Artifact content_type too long: {} > {}",
                    content_type.len(),
                    MAX_ARTIFACT_CONTENT_TYPE_LENGTH
                )));
            }
        }

        if !artifact.metadata.is_empty() {
            let metadata_size = serde_json::to_string(&artifact.metadata)
                .map(|s| s.len())
                .unwrap_or(0);
            if metadata_size > MAX_ARTIFACT_METADATA_BYTES {
                return Err(CoreError::InvalidInput(format!(
                    "Artifact metadata too large: {} > {}",
                    metadata_size, MAX_ARTIFACT_METADATA_BYTES
                )));
            }
        }
    }

    if total_size > MAX_TOTAL_INLINE_ARTIFACTS_BYTES {
        return Err(CoreError::InvalidInput(format!(
            "Total artifacts size {} exceeds {} bytes",
            total_size, MAX_TOTAL_INLINE_ARTIFACTS_BYTES
        )));
    }

    Ok(())
}

pub fn validate_context_overrides(overrides: &[ContextOverride]) -> Result<(), CoreError> {
    if overrides.len() > MAX_CONTEXT_OVERRIDES_PER_ROLLOUT {
        return Err(CoreError::InvalidInput(format!(
            "Too many context overrides: {} > {}",
            overrides.len(),
            MAX_CONTEXT_OVERRIDES_PER_ROLLOUT
        )));
    }

    let mut total_size = 0usize;
    let mut total_files = 0usize;
    for override_item in overrides {
        total_size += override_item.size_bytes();
        total_files += override_item.file_artifacts.len();
    }

    if total_size > MAX_CONTEXT_SNAPSHOT_BYTES {
        return Err(CoreError::InvalidInput(format!(
            "Total context override size {} exceeds {} bytes",
            total_size, MAX_CONTEXT_SNAPSHOT_BYTES
        )));
    }

    let max_files = 50usize;
    if total_files > max_files {
        return Err(CoreError::InvalidInput(format!(
            "Too many file artifacts across overrides: {} > {}",
            total_files, max_files
        )));
    }

    Ok(())
}

pub fn validate_context_snapshot(snapshot_data: &Value) -> Result<(), CoreError> {
    let size = serde_json::to_string(snapshot_data)
        .map(|s| s.len())
        .unwrap_or(0);
    if size > MAX_CONTEXT_SNAPSHOT_BYTES {
        return Err(CoreError::InvalidInput(format!(
            "Context snapshot size {} exceeds {} bytes",
            size, MAX_CONTEXT_SNAPSHOT_BYTES
        )));
    }
    Ok(())
}
