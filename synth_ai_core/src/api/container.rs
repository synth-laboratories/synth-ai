//! Managed Container deployment client.

use std::path::Path;
use std::time::{Duration, Instant};

use flate2::write::GzEncoder;
use flate2::Compression;
use tar::Builder;

use crate::http::{HttpError, MultipartFile};
use crate::CoreError;

use super::client::SynthClient;
use super::types::{
    ContainerDeployResponse, ContainerDeploySpec, ContainerDeployStatus, ContainerDeploymentInfo,
};

const DEPLOYMENTS_ENDPOINT: &str = "/api/container/deployments";

/// Client for managed Container deployment APIs.
pub struct ContainerDeployClient<'a> {
    client: &'a SynthClient,
}

impl<'a> ContainerDeployClient<'a> {
    pub(crate) fn new(client: &'a SynthClient) -> Self {
        Self { client }
    }

    /// Deploy a Container from a context directory.
    pub async fn deploy_from_dir(
        &self,
        spec: ContainerDeploySpec,
        context_dir: impl AsRef<Path>,
        wait_for_ready: bool,
        build_timeout_s: f64,
    ) -> Result<ContainerDeployResponse, CoreError> {
        let archive = package_context(context_dir.as_ref())?;
        self.deploy_from_archive(spec, archive, wait_for_ready, build_timeout_s)
            .await
    }

    /// Deploy a Container from an in-memory archive.
    pub async fn deploy_from_archive(
        &self,
        spec: ContainerDeploySpec,
        archive_bytes: Vec<u8>,
        wait_for_ready: bool,
        build_timeout_s: f64,
    ) -> Result<ContainerDeployResponse, CoreError> {
        let spec_json = serde_json::to_string(&spec)
            .map_err(|e| CoreError::Validation(format!("invalid spec: {}", e)))?;

        let data = vec![("spec_json".to_string(), spec_json)];
        let files = vec![MultipartFile::new(
            "context",
            "context.tar.gz",
            archive_bytes,
            Some("application/gzip".to_string()),
        )];

        let mut response: ContainerDeployResponse = self
            .client
            .http
            .post_multipart(DEPLOYMENTS_ENDPOINT, &data, &files)
            .await
            .map_err(map_http_error)?;

        if wait_for_ready {
            response = self.wait_for_ready(response, build_timeout_s).await?;
        }

        Ok(response)
    }

    /// Fetch a deployment by ID.
    pub async fn get(&self, deployment_id: &str) -> Result<ContainerDeploymentInfo, CoreError> {
        let path = format!("{}/{}", DEPLOYMENTS_ENDPOINT, deployment_id);
        self.client
            .http
            .get(&path, None)
            .await
            .map_err(map_http_error)
    }

    /// List deployments for the current organization.
    pub async fn list(&self) -> Result<Vec<ContainerDeploymentInfo>, CoreError> {
        self.client
            .http
            .get(DEPLOYMENTS_ENDPOINT, None)
            .await
            .map_err(map_http_error)
    }

    /// Fetch deployment status.
    pub async fn status(&self, deployment_id: &str) -> Result<ContainerDeployStatus, CoreError> {
        let path = format!("{}/{}/status", DEPLOYMENTS_ENDPOINT, deployment_id);
        self.client
            .http
            .get(&path, None)
            .await
            .map_err(map_http_error)
    }

    async fn wait_for_ready(
        &self,
        mut response: ContainerDeployResponse,
        build_timeout_s: f64,
    ) -> Result<ContainerDeployResponse, CoreError> {
        let deadline = Instant::now() + Duration::from_secs_f64(build_timeout_s.max(1.0));
        while Instant::now() < deadline {
            let status = self.status(&response.deployment_id).await?;
            response.status = status.status;
            if matches!(response.status.as_str(), "ready" | "failed") {
                return Ok(response);
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        Ok(response)
    }
}

fn package_context(context_dir: &Path) -> Result<Vec<u8>, CoreError> {
    if !context_dir.exists() {
        return Err(CoreError::InvalidInput(format!(
            "context dir not found: {}",
            context_dir.display()
        )));
    }
    if !context_dir.is_dir() {
        return Err(CoreError::InvalidInput(format!(
            "context path is not a directory: {}",
            context_dir.display()
        )));
    }

    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut builder = Builder::new(encoder);
    add_dir_to_archive(&mut builder, context_dir, context_dir)?;
    builder
        .finish()
        .map_err(|e| CoreError::Internal(format!("failed to finish archive: {}", e)))?;
    let encoder = builder
        .into_inner()
        .map_err(|e| CoreError::Internal(format!("failed to finalize archive: {}", e)))?;
    encoder
        .finish()
        .map_err(|e| CoreError::Internal(format!("failed to write archive: {}", e)))
}

fn add_dir_to_archive(
    builder: &mut Builder<GzEncoder<Vec<u8>>>,
    base: &Path,
    dir: &Path,
) -> Result<(), CoreError> {
    for entry in std::fs::read_dir(dir)
        .map_err(|e| CoreError::Internal(format!("failed to read dir: {}", e)))?
    {
        let entry =
            entry.map_err(|e| CoreError::Internal(format!("failed to read entry: {}", e)))?;
        let path = entry.path();
        if path.is_dir() {
            add_dir_to_archive(builder, base, &path)?;
        } else if path.is_file() {
            let rel = path
                .strip_prefix(base)
                .map_err(|e| CoreError::Internal(format!("failed to strip path prefix: {}", e)))?;
            builder.append_path_with_name(&path, rel).map_err(|e| {
                CoreError::Internal(format!("failed to add file to archive: {}", e))
            })?;
        }
    }
    Ok(())
}

fn map_http_error(e: HttpError) -> CoreError {
    match e {
        HttpError::Response(detail) => {
            if detail.status == 401 || detail.status == 403 {
                CoreError::Authentication(format!("authentication failed: {}", detail))
            } else if detail.status == 429 {
                CoreError::UsageLimit(crate::UsageLimitInfo::from_http_429("container", &detail))
            } else {
                CoreError::HttpResponse(crate::HttpErrorInfo {
                    status: detail.status,
                    url: detail.url,
                    message: detail.message,
                    body_snippet: detail.body_snippet,
                })
            }
        }
        HttpError::Request(e) => CoreError::Http(e),
        _ => CoreError::Internal(format!("{}", e)),
    }
}
