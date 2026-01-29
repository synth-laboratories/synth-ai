//! MIPRO-specific helpers.

use crate::api::{MiproJobRequest, SynthClient};
use crate::errors::CoreError;

/// Submit a MIPRO job and return the job ID.
pub async fn submit_mipro_job(
    client: &SynthClient,
    request: MiproJobRequest,
) -> Result<String, CoreError> {
    client.jobs().submit_mipro(request).await
}

