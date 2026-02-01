//! Prompt learning job orchestration.
//!
//! This module provides the high-level `PromptLearningJob` class for
//! submitting and tracking GEPA/MIPRO optimization jobs.

use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::types::PolicyJobStatus;
use crate::api::SynthClient;
use crate::auth;
use crate::errors::CoreError;

use super::events::{ParsedEvent, TerminalStatus};
use super::progress::ProgressTracker;
use crate::sse::stream_sse_events;
use futures_util::StreamExt;

/// Result from a prompt learning job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptLearningResult {
    /// Job ID
    pub job_id: String,
    /// Current status
    pub status: PolicyJobStatus,
    /// Best reward achieved
    #[serde(default, alias = "best_score")]
    pub best_reward: Option<f64>,
    /// Best prompt configuration
    #[serde(default)]
    pub best_prompt: Option<Value>,
    /// Baseline reward
    #[serde(default, alias = "baseline_score")]
    pub baseline_reward: Option<f64>,
    /// Number of candidates evaluated
    #[serde(default)]
    pub candidates_evaluated: i32,
    /// Number of generations completed
    #[serde(default)]
    pub generations_completed: i32,
    /// Error message if failed
    #[serde(default)]
    pub error: Option<String>,
    /// Raw response data
    #[serde(default)]
    pub raw: Value,
}

impl PromptLearningResult {
    /// Check if the job succeeded.
    pub fn succeeded(&self) -> bool {
        self.status == PolicyJobStatus::Succeeded
    }

    /// Check if the job failed.
    pub fn failed(&self) -> bool {
        self.status == PolicyJobStatus::Failed
    }

    /// Check if the job is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    /// Get the system prompt from best_prompt if available.
    pub fn get_system_prompt(&self) -> Option<String> {
        self.best_prompt.as_ref().and_then(|p| {
            // Try various paths where system prompt might be
            p.get("system_prompt")
                .and_then(|v| v.as_str())
                .or_else(|| p.get("instruction").and_then(|v| v.as_str()))
                .or_else(|| {
                    p.get("stages")
                        .and_then(|s| s.get("main"))
                        .and_then(|m| m.get("instruction"))
                        .and_then(|v| v.as_str())
                })
                .map(|s| s.to_string())
        })
    }
}

/// Ranked prompt from results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedPrompt {
    /// Rank (1 = best)
    pub rank: i32,
    /// Candidate ID
    pub candidate_id: String,
    /// Training accuracy
    #[serde(default)]
    pub train_accuracy: Option<f64>,
    /// Validation accuracy
    #[serde(default)]
    pub val_accuracy: Option<f64>,
    /// Prompt text or configuration
    #[serde(default)]
    pub prompt: Option<Value>,
}

/// Extracted prompt results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptResults {
    /// Best prompt text
    #[serde(default)]
    pub best_prompt: Option<String>,
    /// Best reward
    #[serde(default, alias = "best_score")]
    pub best_reward: Option<f64>,
    /// Top prompts ranked by score
    #[serde(default)]
    pub top_prompts: Vec<RankedPrompt>,
}

/// High-level prompt learning job orchestration.
pub struct PromptLearningJob {
    /// Synth API client
    client: SynthClient,
    /// Job ID (set after submit)
    job_id: Option<String>,
    /// Job configuration
    config: Value,
    /// Optional SynthTunnel worker token
    task_app_worker_token: Option<String>,
    /// Progress tracker
    tracker: ProgressTracker,
}

impl PromptLearningJob {
    /// Create a job from a configuration dict.
    ///
    /// # Arguments
    ///
    /// * `config` - Job configuration (algorithm, task_app_url, policy, etc.)
    /// * `api_key` - Optional API key (uses env if not provided)
    /// * `base_url` - Optional base URL (uses default if not provided)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let job = PromptLearningJob::from_dict(
    ///     serde_json::json!({
    ///         "algorithm": "gepa",
    ///         "task_app_url": "http://localhost:8000",
    ///         "env_name": "default",
    ///         "policy": { "model": "gpt-4o-mini", "provider": "openai" },
    ///         "gepa": { "rollout_budget": 100 }
    ///     }),
    ///     None,
    ///     None,
    ///     None,
    /// )?;
    /// ```
    pub fn from_dict(
        config: Value,
        api_key: Option<&str>,
        base_url: Option<&str>,
        task_app_worker_token: Option<String>,
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
            config,
            task_app_worker_token,
            tracker: ProgressTracker::new(),
        })
    }

    /// Reconnect to an existing job by ID.
    ///
    /// # Arguments
    ///
    /// * `job_id` - Existing job ID
    /// * `api_key` - Optional API key
    /// * `base_url` - Optional base URL
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
            config: Value::Null,
            task_app_worker_token: None,
            tracker: ProgressTracker::new(),
        })
    }

    /// Get the job ID (if submitted).
    pub fn job_id(&self) -> Option<&str> {
        self.job_id.as_deref()
    }

    /// Get the progress tracker.
    pub fn tracker(&self) -> &ProgressTracker {
        &self.tracker
    }

    /// Submit the job to the backend.
    ///
    /// Returns the job ID on success.
    pub async fn submit(&mut self) -> Result<String, CoreError> {
        if self.job_id.is_some() {
            return Err(CoreError::Validation("job already submitted".to_string()));
        }

        if self.config.is_null() {
            return Err(CoreError::Validation(
                "no configuration provided".to_string(),
            ));
        }

        // Submit via jobs API
        let job_id = self
            .client
            .jobs()
            .submit_raw_with_worker_token(self.config.clone(), self.task_app_worker_token.clone())
            .await?;
        self.job_id = Some(job_id.clone());

        Ok(job_id)
    }

    /// Get the current job status.
    pub async fn get_status(&self) -> Result<PromptLearningResult, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;

        let result = self.client.jobs().get_status(job_id).await?;

        Ok(PromptLearningResult {
            job_id: result.job_id,
            status: result.status,
            best_reward: result.best_reward,
            best_prompt: result.best_prompt,
            baseline_reward: None,
            candidates_evaluated: result.candidates_evaluated.unwrap_or(0),
            generations_completed: result.generations_completed.unwrap_or(0),
            error: result.error,
            raw: Value::Null,
        })
    }

    /// Poll until the job reaches a terminal state.
    ///
    /// # Arguments
    ///
    /// * `timeout_secs` - Maximum time to wait
    /// * `interval_secs` - Polling interval
    pub async fn poll_until_complete(
        &self,
        timeout_secs: f64,
        interval_secs: f64,
    ) -> Result<PromptLearningResult, CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;

        let result = self
            .client
            .jobs()
            .poll_until_complete(job_id, timeout_secs, interval_secs)
            .await?;

        Ok(PromptLearningResult {
            job_id: result.job_id,
            status: result.status,
            best_reward: result.best_reward,
            best_prompt: result.best_prompt,
            baseline_reward: None,
            candidates_evaluated: result.candidates_evaluated.unwrap_or(0),
            generations_completed: result.generations_completed.unwrap_or(0),
            error: result.error,
            raw: Value::Null,
        })
    }

    /// Stream events until completion with callback.
    ///
    /// # Arguments
    ///
    /// * `timeout_secs` - Maximum time to wait
    /// * `on_event` - Optional callback for each event
    pub async fn stream_until_complete<F>(
        &mut self,
        timeout_secs: f64,
        mut on_event: Option<F>,
    ) -> Result<PromptLearningResult, CoreError>
    where
        F: FnMut(&ParsedEvent),
    {
        use std::cell::Cell;

        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;

        eprintln!(
            "[PL] stream_until_complete: job={} timeout={:.0}s",
            job_id, timeout_secs
        );

        let timeout = Duration::from_secs_f64(timeout_secs);
        let base_url = self.client.base_url().trim_end_matches('/').to_string();
        let events_url = format!(
            "{}/api/prompt-learning/online/jobs/{}/events/stream",
            base_url, job_id
        );
        let api_key = self.client.http().api_key().to_string();
        let mut headers = HeaderMap::new();
        headers.insert("Accept", HeaderValue::from_static("text/event-stream"));
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|_| CoreError::Validation("invalid api key".to_string()))?,
        );
        headers.insert(
            "X-API-Key",
            HeaderValue::from_str(&api_key)
                .map_err(|_| CoreError::Validation("invalid api key".to_string()))?,
        );

        // Use Cell for interior mutability to satisfy borrow checker
        let terminal_reached = Cell::new(false);
        let event_count = Cell::new(0u64);
        let terminal_status = Cell::new(None);

        {
            let tracker = &mut self.tracker;

            let mut stream =
                stream_sse_events(&events_url, "GET", headers, None, Some(timeout)).await?;

            while let Some(item) = stream.next().await {
                let event = item?;
                if event.data.trim() == "[DONE]" {
                    break;
                }

                let payload: Value = serde_json::from_str(&event.data).unwrap_or(Value::Null);
                let parsed = super::events::EventParser::parse(&payload);
                let count = event_count.get() + 1;
                event_count.set(count);

                // Update tracker
                tracker.update(&parsed);

                // Log progress periodically
                if count % 5 == 0 || parsed.category.is_terminal() {
                    eprintln!(
                        "[PL] Event #{}: type={} category={:?} | tracker: best={:.3} baseline={:?} candidates={} gens={}",
                        count,
                        parsed.event_type,
                        parsed.category,
                        tracker.best_reward(),
                        tracker.baseline_reward(),
                        tracker.progress.candidates_evaluated,
                        tracker.progress.generations_completed,
                    );
                }

                // Call user callback
                if let Some(ref mut cb) = on_event {
                    cb(&parsed);
                }

                // Check for terminal
                if parsed.category.is_terminal() {
                    eprintln!(
                        "[PL] Terminal event received: {} (category={:?})",
                        parsed.event_type, parsed.category
                    );
                    terminal_status.set(super::events::EventParser::terminal_status(
                        &parsed.event_type,
                    ));
                    terminal_reached.set(true);
                    break;
                }
            }
        }

        eprintln!(
            "[PL] stream_until_complete: streaming finished, processed {} events",
            event_count.get()
        );

        if !terminal_reached.get() {
            return Err(CoreError::Timeout(
                "stream ended without terminal event".to_string(),
            ));
        }

        // Get final status (tracker borrow is dropped now)
        eprintln!("[PL] Fetching final job status...");
        let status_result = match self.get_status().await {
            Ok(result) => Some(result),
            Err(err) => {
                eprintln!("[PL] Warning: failed to fetch final job status: {}", err);
                None
            }
        };

        let mut final_status = status_result
            .as_ref()
            .map(|result| result.status)
            .unwrap_or(crate::api::types::PolicyJobStatus::Succeeded);
        if !final_status.is_terminal() {
            if let Some(status) = terminal_status.get() {
                final_status = match status {
                    TerminalStatus::Succeeded => crate::api::types::PolicyJobStatus::Succeeded,
                    TerminalStatus::Failed => crate::api::types::PolicyJobStatus::Failed,
                    TerminalStatus::Cancelled => crate::api::types::PolicyJobStatus::Cancelled,
                };
                eprintln!(
                    "[PL] Final status override from terminal event: {:?}",
                    final_status
                );
            }
        }
        eprintln!(
            "[PL] Final status: status={:?} best_reward={:?} error={:?}",
            final_status,
            status_result.as_ref().and_then(|result| result.best_reward),
            status_result
                .as_ref()
                .and_then(|result| result.error.clone())
        );

        // Merge tracker data with status (fall back to tracker if status fetch failed)
        let result = PromptLearningResult {
            job_id: status_result
                .as_ref()
                .map(|result| result.job_id.clone())
                .unwrap_or_else(|| job_id.to_string()),
            status: final_status,
            best_reward: status_result
                .as_ref()
                .and_then(|result| result.best_reward)
                .or(Some(self.tracker.best_reward())),
            best_prompt: status_result
                .as_ref()
                .and_then(|result| result.best_prompt.clone()),
            baseline_reward: self.tracker.baseline_reward(),
            candidates_evaluated: self.tracker.progress.candidates_evaluated,
            generations_completed: self.tracker.progress.generations_completed,
            error: status_result.and_then(|result| result.error),
            raw: Value::Null,
        };

        eprintln!(
            "[PL] RESULT: status={:?} best={:?} baseline={:?} candidates={} gens={}",
            result.status,
            result.best_reward,
            result.baseline_reward,
            result.candidates_evaluated,
            result.generations_completed
        );

        Ok(result)
    }

    /// Cancel a running job.
    ///
    /// # Arguments
    ///
    /// * `reason` - Optional cancellation reason
    pub async fn cancel(&self, reason: Option<&str>) -> Result<(), CoreError> {
        let job_id = self
            .job_id
            .as_ref()
            .ok_or_else(|| CoreError::Validation("job not submitted yet".to_string()))?;

        self.client.jobs().cancel(job_id, reason).await
    }

    /// Get detailed results including prompt extraction.
    ///
    /// This fetches events to extract the best prompts.
    pub async fn get_results(&self) -> Result<PromptResults, CoreError> {
        // Get final status for best_prompt
        let status = self.get_status().await?;

        let best_prompt = status.get_system_prompt();
        let best_reward = status.best_reward.or(Some(self.tracker.best_reward()));

        // Build ranked prompts from tracker candidates
        let mut top_prompts: Vec<RankedPrompt> = self
            .tracker
            .candidates
            .iter()
            .filter(|c| c.accepted || c.is_pareto)
            .map(|c| RankedPrompt {
                rank: 0,
                candidate_id: c.candidate_id.clone(),
                train_accuracy: c.reward,
                val_accuracy: c.val_reward,
                prompt: None,
            })
            .collect();

        // Sort by accuracy descending
        top_prompts.sort_by(|a, b| {
            let a_score = a.val_accuracy.or(a.train_accuracy).unwrap_or(0.0);
            let b_score = b.val_accuracy.or(b.train_accuracy).unwrap_or(0.0);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, prompt) in top_prompts.iter_mut().enumerate() {
            prompt.rank = (i + 1) as i32;
        }

        Ok(PromptResults {
            best_prompt,
            best_reward,
            top_prompts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_result_status() {
        let result = PromptLearningResult {
            job_id: "test".to_string(),
            status: PolicyJobStatus::Succeeded,
            best_reward: Some(0.85),
            best_prompt: None,
            baseline_reward: None,
            candidates_evaluated: 10,
            generations_completed: 3,
            error: None,
            raw: Value::Null,
        };

        assert!(result.succeeded());
        assert!(!result.failed());
        assert!(result.is_terminal());
    }

    #[test]
    fn test_result_get_system_prompt() {
        let result = PromptLearningResult {
            job_id: "test".to_string(),
            status: PolicyJobStatus::Succeeded,
            best_reward: Some(0.85),
            best_prompt: Some(json!({
                "system_prompt": "You are a helpful assistant."
            })),
            baseline_reward: None,
            candidates_evaluated: 10,
            generations_completed: 3,
            error: None,
            raw: Value::Null,
        };

        assert_eq!(
            result.get_system_prompt(),
            Some("You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_ranked_prompt_sorting() {
        let mut prompts = vec![
            RankedPrompt {
                rank: 0,
                candidate_id: "a".to_string(),
                train_accuracy: Some(0.7),
                val_accuracy: None,
                prompt: None,
            },
            RankedPrompt {
                rank: 0,
                candidate_id: "b".to_string(),
                train_accuracy: Some(0.9),
                val_accuracy: None,
                prompt: None,
            },
            RankedPrompt {
                rank: 0,
                candidate_id: "c".to_string(),
                train_accuracy: Some(0.8),
                val_accuracy: Some(0.85),
                prompt: None,
            },
        ];

        // Sort by accuracy descending (val_accuracy takes precedence)
        prompts.sort_by(|a, b| {
            let a_score = a.val_accuracy.or(a.train_accuracy).unwrap_or(0.0);
            let b_score = b.val_accuracy.or(b.train_accuracy).unwrap_or(0.0);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        assert_eq!(prompts[0].candidate_id, "b"); // 0.9
        assert_eq!(prompts[1].candidate_id, "c"); // 0.85 (val)
        assert_eq!(prompts[2].candidate_id, "a"); // 0.7
    }
}
