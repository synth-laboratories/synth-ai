//! Graph Evolve job orchestration.
//!
//! Provides a higher-level interface for submitting and tracking Graph Evolve jobs.

use serde_json::Value;

use crate::api::SynthClient;
use crate::auth;
use crate::errors::CoreError;

/// High-level Graph Evolve job handle.
pub struct GraphEvolveJob {
    client: SynthClient,
    job_id: Option<String>,
    payload: Option<Value>,
}

impl GraphEvolveJob {
    /// Create a job from a payload dictionary.
    pub fn from_payload(
        payload: Value,
        api_key: Option<&str>,
        base_url: Option<&str>,
    ) -> Result<Self, CoreError> {
        let api_key = match api_key {
            Some(k) => k.to_string(),
            None => auth::get_api_key(None)
                .ok_or_else(|| CoreError::Authentication("SYNTH_API_KEY not found".to_string()))?,
        };
        let client = SynthClient::new(&api_key, base_url)?;
        Ok(Self {
            client,
            job_id: None,
            payload: Some(payload),
        })
    }

    /// Reconnect to an existing job by ID.
    pub fn from_job_id(
        job_id: &str,
        api_key: Option<&str>,
        base_url: Option<&str>,
    ) -> Result<Self, CoreError> {
        let api_key = match api_key {
            Some(k) => k.to_string(),
            None => auth::get_api_key(None)
                .ok_or_else(|| CoreError::Authentication("SYNTH_API_KEY not found".to_string()))?,
        };
        let client = SynthClient::new(&api_key, base_url)?;
        Ok(Self {
            client,
            job_id: Some(job_id.to_string()),
            payload: None,
        })
    }

    /// Get the job ID if available.
    pub fn job_id(&self) -> Option<&str> {
        self.job_id.as_deref()
    }

    /// Submit the job to the backend.
    pub async fn submit(&mut self) -> Result<String, CoreError> {
        if self.job_id.is_some() {
            return Err(CoreError::Validation("job already submitted".to_string()));
        }
        let payload = self
            .payload
            .as_ref()
            .ok_or_else(|| CoreError::Validation("no payload provided".to_string()))?
            .clone();

        let response = self.client.graph_evolve().submit_job(payload).await?;
        let job_id = response
            .get("job_id")
            .and_then(|v| v.as_str())
            .or_else(|| response.get("graph_evolve_job_id").and_then(|v| v.as_str()))
            .ok_or_else(|| CoreError::Validation("response missing job_id".to_string()))?
            .to_string();
        self.job_id = Some(job_id.clone());
        Ok(job_id)
    }

    /// Fetch job status.
    pub async fn get_status(&self) -> Result<Value, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;
        self.client.graph_evolve().get_status(job_id).await
    }

    /// Start a queued job.
    pub async fn start(&self) -> Result<Value, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;
        self.client.graph_evolve().start_job(job_id).await
    }

    /// Fetch job events.
    pub async fn get_events(&self, since_seq: i64, limit: i64) -> Result<Value, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;
        self.client
            .graph_evolve()
            .get_events(job_id, since_seq, limit)
            .await
    }

    /// Fetch job metrics.
    pub async fn get_metrics(&self, query: &str) -> Result<Value, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;
        self.client.graph_evolve().get_metrics(job_id, query).await
    }

    /// Download prompt artifacts.
    pub async fn download_prompt(&self) -> Result<Value, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;
        self.client.graph_evolve().download_prompt(job_id).await
    }

    /// Download graph text export.
    pub async fn download_graph_txt(&self) -> Result<String, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;
        self.client.graph_evolve().download_graph_txt(job_id).await
    }

    /// Run inference against the optimized graph.
    pub async fn run_inference(&self, payload: Value) -> Result<Value, CoreError> {
        self.client.graph_evolve().run_inference(payload).await
    }

    /// Fetch a graph record snapshot.
    pub async fn get_graph_record(&self, payload: Value) -> Result<Value, CoreError> {
        self.client.graph_evolve().get_graph_record(payload).await
    }

    /// Query the workflow state.
    pub async fn query_workflow_state(&self) -> Result<Value, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;
        self.client.graph_evolve().query_workflow_state(job_id).await
    }

    /// Cancel a running job.
    pub async fn cancel(&self, reason: Option<String>) -> Result<Value, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;
        let mut payload = serde_json::Map::new();
        if let Some(reason) = reason {
            payload.insert("reason".to_string(), Value::String(reason));
        }
        self.client
            .graph_evolve()
            .cancel_job(job_id, Value::Object(payload))
            .await
    }
}
