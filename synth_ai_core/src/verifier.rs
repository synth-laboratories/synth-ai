//! Verifier helpers built on the Graphs API.

use serde_json::Value;

use crate::api::{SynthClient, VerifierOptions, VerifierResponse};
use crate::errors::CoreError;

/// Verifier client wrapper.
pub struct VerifierClient<'a> {
    client: &'a SynthClient,
}

impl<'a> VerifierClient<'a> {
    /// Create a new verifier client.
    pub fn new(client: &'a SynthClient) -> Self {
        Self { client }
    }

    /// Verify a trace against a rubric using the graphs API.
    pub async fn verify(
        &self,
        trace: Value,
        rubric: Value,
        options: Option<VerifierOptions>,
    ) -> Result<VerifierResponse, CoreError> {
        self.client.graphs().verify(trace, rubric, options).await
    }
}

