//! Event streaming for optimization jobs.
//!
//! This module provides SSE-style event streaming with deduplication
//! for optimization jobs.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::errors::CoreError;
use crate::http::HttpClient;

use super::events::{EventCategory, EventParser, ParsedEvent};

/// Format a duration for human-readable logging.
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m{}s", secs / 60, secs % 60)
    } else {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Log an event summary based on its category.
pub fn log_event_summary(event: &ParsedEvent) {
    let path = EventParser::parse_path(&event.event_type);
    if let (Some(entity), Some(action)) = (path.entity.as_deref(), path.action.as_deref()) {
        eprintln!(
            "[STREAM] Event path: {}.{} (alg={:?} detail={:?})",
            entity, action, path.algorithm, path.detail
        );
    }
    match event.category {
        EventCategory::Baseline => {
            let baseline = EventParser::parse_baseline(event);
            eprintln!("[STREAM] Baseline: reward={:.3?}", baseline.reward);
        }
        EventCategory::Candidate => {
            let candidate = EventParser::parse_candidate(event);
            eprintln!(
                "[STREAM] Candidate {}: reward={:.3?} accepted={} gen={:?}",
                candidate.candidate_id, candidate.reward, candidate.accepted, candidate.generation
            );
        }
        EventCategory::Frontier => {
            let frontier = EventParser::parse_frontier(event);
            eprintln!(
                "[STREAM] Frontier updated: size={} best={:.3?}",
                frontier.frontier_size, frontier.best_reward
            );
        }
        EventCategory::Progress => {
            let progress = EventParser::parse_progress(event);
            eprintln!(
                "[STREAM] Progress: rollouts={}/{:?} best={:.3?}",
                progress.rollouts_completed, progress.rollouts_total, progress.best_reward
            );
        }
        EventCategory::Generation => {
            let gen = EventParser::parse_generation(event);
            eprintln!(
                "[STREAM] Generation {}: best_acc={:.3} proposed={} accepted={}",
                gen.generation, gen.best_reward, gen.candidates_proposed, gen.candidates_accepted
            );
        }
        EventCategory::Validation => {
            eprintln!("[STREAM] Validation event: {:?}", event.event_type);
        }
        EventCategory::Complete => {
            let complete = EventParser::parse_complete(event);
            eprintln!(
                "[STREAM] COMPLETE: best={:.3?} baseline={:.3?} reason={:?}",
                complete.best_reward, complete.baseline_reward, complete.finish_reason
            );
        }
        EventCategory::Termination => {
            let term = EventParser::parse_termination(event);
            eprintln!("[STREAM] TERMINATION: reason={}", term.reason);
        }
        EventCategory::Usage => {
            let usage = EventParser::parse_usage(event);
            eprintln!(
                "[STREAM] Usage: total=${:.4} tokens=${:.4} sandbox=${:.4}",
                usage.total_usd, usage.tokens_usd, usage.sandbox_usd
            );
        }
        EventCategory::Throughput => {
            eprintln!("[STREAM] Throughput event");
        }
        EventCategory::Unknown => {
            eprintln!("[STREAM] Unknown event: {}", event.event_type);
        }
    }
}

/// Event stream for polling job events.
pub struct EventStream {
    /// HTTP client reference
    client: HttpClient,
    /// Job ID to stream events for
    job_id: String,
    /// Base URL for API
    base_url: String,
    /// Last seen sequence number
    last_seq: i64,
    /// Whether to deduplicate events
    deduplicate: bool,
    /// Set of seen sequence numbers
    seen_seqs: HashSet<i64>,
    /// Maximum events per poll
    max_events_per_poll: i32,
}

impl EventStream {
    /// Create a new event stream for a job.
    pub fn new(client: HttpClient, base_url: &str, job_id: &str) -> Self {
        Self {
            client,
            job_id: job_id.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            last_seq: 0,
            deduplicate: true,
            seen_seqs: HashSet::new(),
            max_events_per_poll: 500,
        }
    }

    /// Set the starting sequence number.
    pub fn with_start_seq(mut self, seq: i64) -> Self {
        self.last_seq = seq;
        self
    }

    /// Enable or disable deduplication.
    pub fn with_deduplicate(mut self, dedupe: bool) -> Self {
        self.deduplicate = dedupe;
        self
    }

    /// Set max events per poll.
    pub fn with_max_events(mut self, max: i32) -> Self {
        self.max_events_per_poll = max;
        self
    }

    /// Get the last seen sequence number.
    pub fn last_seq(&self) -> i64 {
        self.last_seq
    }

    /// Poll for new events.
    ///
    /// Returns events since the last sequence number.
    pub async fn poll_events(&mut self) -> Result<Vec<ParsedEvent>, CoreError> {
        let url = format!(
            "{}/api/prompt-learning/online/jobs/{}/events",
            self.base_url, self.job_id
        );

        let params = [
            ("since_seq", self.last_seq.to_string()),
            ("limit", self.max_events_per_poll.to_string()),
        ];

        let params_slice: &[(&str, &str)] = &[("since_seq", &params[0].1), ("limit", &params[1].1)];

        eprintln!(
            "[STREAM] poll_events: job={} since_seq={} limit={}",
            self.job_id, self.last_seq, self.max_events_per_poll
        );

        let response: Value = self
            .client
            .get(&url, Some(params_slice))
            .await
            .map_err(|e| {
                eprintln!("[STREAM] ERROR: poll_events failed: {}", e);
                CoreError::Internal(format!("failed to fetch events: {}", e))
            })?;

        // Parse events array
        let events_array = response
            .get("events")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        eprintln!(
            "[STREAM] poll_events: received {} raw events",
            events_array.len()
        );

        let mut parsed_events = Vec::new();

        for event_value in events_array {
            let parsed = EventParser::parse(&event_value);

            // Update last_seq
            if let Some(seq) = parsed.seq {
                if seq > self.last_seq {
                    self.last_seq = seq;
                }

                // Deduplication
                if self.deduplicate {
                    if self.seen_seqs.contains(&seq) {
                        continue;
                    }
                    self.seen_seqs.insert(seq);

                    // Limit seen_seqs size to prevent memory growth
                    if self.seen_seqs.len() > 10000 {
                        // Keep only recent sequences
                        let threshold = self.last_seq - 5000;
                        self.seen_seqs.retain(|&s| s > threshold);
                    }
                }
            }

            parsed_events.push(parsed);
        }

        if !parsed_events.is_empty() {
            eprintln!(
                "[STREAM] poll_events: returning {} new events (last_seq={})",
                parsed_events.len(),
                self.last_seq
            );
        }

        Ok(parsed_events)
    }

    /// Stream events until a terminal condition with callback.
    ///
    /// # Arguments
    ///
    /// * `on_event` - Callback for each event
    /// * `timeout` - Maximum time to stream
    /// * `poll_interval` - Time between polls
    /// * `is_terminal` - Optional check for terminal status
    pub async fn stream_until<F, T>(
        &mut self,
        mut on_event: F,
        timeout: Duration,
        poll_interval: Duration,
        mut is_terminal: T,
    ) -> Result<(), CoreError>
    where
        F: FnMut(&ParsedEvent),
        T: FnMut() -> bool,
    {
        let start = Instant::now();
        let mut last_event_time = Instant::now();
        let mut poll_count = 0u64;
        let mut total_events = 0u64;

        eprintln!(
            "[STREAM] stream_until: starting job={} timeout={} poll_interval={}",
            self.job_id,
            format_duration(timeout),
            format_duration(poll_interval)
        );

        loop {
            let elapsed = start.elapsed();

            // Check timeout
            if elapsed > timeout {
                eprintln!(
                    "[STREAM] TIMEOUT: elapsed={} total_events={}",
                    format_duration(elapsed),
                    total_events
                );
                return Err(CoreError::Timeout(format!(
                    "event stream timed out after {:.0} seconds",
                    timeout.as_secs_f64()
                )));
            }

            // Check terminal condition
            if is_terminal() {
                eprintln!(
                    "[STREAM] Terminal condition reached: elapsed={} total_events={}",
                    format_duration(elapsed),
                    total_events
                );
                return Ok(());
            }

            poll_count += 1;

            // Log every 10 polls or when significant time has passed
            if poll_count % 10 == 0 {
                eprintln!(
                    "[STREAM] Streaming: elapsed={} polls={} events={}",
                    format_duration(elapsed),
                    poll_count,
                    total_events
                );
            }

            // Poll events
            match self.poll_events().await {
                Ok(events) => {
                    if !events.is_empty() {
                        last_event_time = Instant::now();
                        total_events += events.len() as u64;
                        eprintln!(
                            "[STREAM] Received {} events (total={})",
                            events.len(),
                            total_events
                        );
                    }

                    for event in &events {
                        // Log each event summary
                        log_event_summary(event);

                        on_event(event);

                        // Check for terminal events
                        if event.category.is_terminal() {
                            eprintln!(
                                "[STREAM] Terminal event received: {} (elapsed={})",
                                event.event_type,
                                format_duration(elapsed)
                            );
                            return Ok(());
                        }
                    }
                }
                Err(e) => {
                    let since_last = last_event_time.elapsed();
                    eprintln!(
                        "[STREAM] Poll error ({}s since last event): {}",
                        since_last.as_secs(),
                        e
                    );
                    // Allow some grace period for transient errors
                    if since_last > Duration::from_secs(120) {
                        eprintln!("[STREAM] ERROR: Too long since last event, giving up");
                        return Err(e);
                    }
                }
            }

            // Wait before next poll
            tokio::time::sleep(poll_interval).await;
        }
    }
}

/// Stream configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Poll interval in seconds
    pub poll_interval_secs: f64,
    /// Maximum events per poll
    pub max_events_per_poll: i32,
    /// Whether to deduplicate events
    pub deduplicate: bool,
    /// Timeout in seconds
    pub timeout_secs: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: 5.0,
            max_events_per_poll: 500,
            deduplicate: true,
            timeout_secs: 3600.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.poll_interval_secs, 5.0);
        assert_eq!(config.max_events_per_poll, 500);
        assert!(config.deduplicate);
    }
}
