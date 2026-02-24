use std::sync::Arc;

use crate::graphs_api::{GraphCompletionRequest, GraphCompletionResponse};
use crate::openapi_paths;
use crate::transport::Transport;
use crate::types::Result;

#[derive(Clone)]
pub struct VerifiersClient {
    transport: Arc<Transport>,
}

impl VerifiersClient {
    pub(crate) fn new(transport: Arc<Transport>) -> Self {
        Self { transport }
    }

    pub async fn verify(&self, request: &GraphCompletionRequest) -> Result<GraphCompletionResponse> {
        self.transport
            .post_json(openapi_paths::API_GRAPHS_COMPLETIONS, request)
            .await
    }
}
