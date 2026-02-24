//! Stream endpoint configuration for different job types.

/// Endpoint configuration for streaming from a job.
#[derive(Debug, Clone)]
pub struct StreamEndpoints {
    /// Status endpoint.
    pub status: Option<String>,
    /// Events endpoint.
    pub events: Option<String>,
    /// Metrics endpoint.
    pub metrics: Option<String>,
    /// Timeline endpoint.
    pub timeline: Option<String>,
    /// Strict status endpoints (tried in order if primary fails).
    pub status_stricts: Vec<String>,
    /// Strict event endpoints.
    pub event_stricts: Vec<String>,
    /// Strict metrics endpoints.
    pub metric_stricts: Vec<String>,
    /// Strict timeline endpoints.
    pub timeline_stricts: Vec<String>,
}

impl StreamEndpoints {
    /// Create endpoints for a generic learning job.
    pub fn learning(job_id: &str) -> Self {
        let base = format!("/learning/jobs/{}", job_id);
        Self {
            status: Some(base.clone()),
            events: Some(format!("{}/events", base)),
            metrics: Some(format!("{}/metrics", base)),
            timeline: Some(format!("{}/timeline", base)),
            status_stricts: vec![],
            event_stricts: vec![],
            metric_stricts: vec![],
            timeline_stricts: vec![],
        }
    }

    /// Create endpoints for a prompt learning (GEPA) job.
    pub fn prompt_learning(job_id: &str) -> Self {
        let base = format!("/api/v1/offline/jobs/{}", job_id);
        Self {
            status: Some(base.clone()),
            events: Some(format!("{}/events", base)),
            metrics: Some(format!("{}/metrics", base)),
            timeline: None,
            status_stricts: vec![],
            event_stricts: vec![],
            metric_stricts: vec![],
            timeline_stricts: vec![],
        }
    }

    /// Create endpoints for an eval job.
    pub fn eval(job_id: &str) -> Self {
        let base = format!("/api/v1/offline/jobs/{}", job_id);
        Self {
            status: Some(base.clone()),
            events: Some(format!("{}/events", base)),
            metrics: None,
            timeline: None,
            status_stricts: vec![],
            event_stricts: vec![],
            metric_stricts: vec![],
            timeline_stricts: vec![],
        }
    }

    /// Create endpoints for an RL job.
    pub fn rl(job_id: &str) -> Self {
        let base = format!("/rl/jobs/{}", job_id);
        Self {
            status: Some(base.clone()),
            events: Some(format!("{}/events", base)),
            metrics: Some(format!("{}/metrics", base)),
            timeline: Some(format!("{}/timeline", base)),
            status_stricts: vec![
                format!("/learning/jobs/{}", job_id),
                format!("/orchestration/jobs/{}", job_id),
            ],
            event_stricts: vec![
                format!("/learning/jobs/{}/events", job_id),
                format!("/orchestration/jobs/{}/events", job_id),
            ],
            metric_stricts: vec![format!("/learning/jobs/{}/metrics", job_id)],
            timeline_stricts: vec![format!("/learning/jobs/{}/timeline", job_id)],
        }
    }

    /// Create endpoints for an SFT job.
    pub fn sft(job_id: &str) -> Self {
        let base = format!("/sft/jobs/{}", job_id);
        Self {
            status: Some(base.clone()),
            events: Some(format!("{}/events", base)),
            metrics: Some(format!("{}/metrics", base)),
            timeline: None,
            status_stricts: vec![],
            event_stricts: vec![],
            metric_stricts: vec![],
            timeline_stricts: vec![],
        }
    }

    /// Create endpoints for graph optimization.
    pub fn graph_optimization(job_id: &str) -> Self {
        let base = format!("/graphs/optimization/jobs/{}", job_id);
        Self {
            status: Some(base.clone()),
            events: Some(format!("{}/events", base)),
            metrics: Some(format!("{}/metrics", base)),
            timeline: None,
            status_stricts: vec![],
            event_stricts: vec![],
            metric_stricts: vec![],
            timeline_stricts: vec![],
        }
    }

    /// Create endpoints for graph evolve jobs.
    pub fn graph_evolve(job_id: &str) -> Self {
        let base = format!("/graph-evolve/jobs/{}", job_id);
        Self {
            status: Some(base.clone()),
            events: Some(format!("{}/events", base)),
            metrics: Some(format!("{}/metrics", base)),
            timeline: None,
            status_stricts: vec![format!("/graphgen/jobs/{}", job_id)],
            event_stricts: vec![format!("/graphgen/jobs/{}/events", job_id)],
            metric_stricts: vec![format!("/graphgen/jobs/{}/metrics", job_id)],
            timeline_stricts: vec![],
        }
    }

    /// Canonical alias for graph evolve endpoints.
    pub fn graphgen(job_id: &str) -> Self {
        Self::graph_evolve(job_id)
    }

    /// Create custom endpoints.
    pub fn custom(
        status: Option<String>,
        events: Option<String>,
        metrics: Option<String>,
        timeline: Option<String>,
    ) -> Self {
        Self {
            status,
            events,
            metrics,
            timeline,
            status_stricts: vec![],
            event_stricts: vec![],
            metric_stricts: vec![],
            timeline_stricts: vec![],
        }
    }

    /// Add a status strict endpoint.
    pub fn with_status_strict(mut self, endpoint: impl Into<String>) -> Self {
        self.status_stricts.push(endpoint.into());
        self
    }

    /// Add an event strict endpoint.
    pub fn with_event_strict(mut self, endpoint: impl Into<String>) -> Self {
        self.event_stricts.push(endpoint.into());
        self
    }

    /// Add a metrics strict endpoint.
    pub fn with_metric_strict(mut self, endpoint: impl Into<String>) -> Self {
        self.metric_stricts.push(endpoint.into());
        self
    }

    /// Add a timeline strict endpoint.
    pub fn with_timeline_strict(mut self, endpoint: impl Into<String>) -> Self {
        self.timeline_stricts.push(endpoint.into());
        self
    }

    /// Get the SSE stream URL for events.
    pub fn events_stream_url(&self) -> Option<String> {
        self.events.as_ref().map(|e| format!("{}/stream", e))
    }

    /// Get all status endpoints to try (primary + strict_paths).
    pub fn all_status_endpoints(&self) -> Vec<&str> {
        let mut endpoints = Vec::new();
        if let Some(ref s) = self.status {
            endpoints.push(s.as_str());
        }
        for strict in &self.status_stricts {
            endpoints.push(strict.as_str());
        }
        endpoints
    }

    /// Get all event endpoints to try (primary + strict_paths).
    pub fn all_event_endpoints(&self) -> Vec<&str> {
        let mut endpoints = Vec::new();
        if let Some(ref e) = self.events {
            endpoints.push(e.as_str());
        }
        for strict in &self.event_stricts {
            endpoints.push(strict.as_str());
        }
        endpoints
    }

    /// Get all metrics endpoints to try (primary + strict_paths).
    pub fn all_metric_endpoints(&self) -> Vec<&str> {
        let mut endpoints = Vec::new();
        if let Some(ref m) = self.metrics {
            endpoints.push(m.as_str());
        }
        for strict in &self.metric_stricts {
            endpoints.push(strict.as_str());
        }
        endpoints
    }

    /// Get all timeline endpoints to try (primary + strict_paths).
    pub fn all_timeline_endpoints(&self) -> Vec<&str> {
        let mut endpoints = Vec::new();
        if let Some(ref t) = self.timeline {
            endpoints.push(t.as_str());
        }
        for strict in &self.timeline_stricts {
            endpoints.push(strict.as_str());
        }
        endpoints
    }
}

impl Default for StreamEndpoints {
    fn default() -> Self {
        Self {
            status: None,
            events: None,
            metrics: None,
            timeline: None,
            status_stricts: vec![],
            event_stricts: vec![],
            metric_stricts: vec![],
            timeline_stricts: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_endpoints() {
        let endpoints = StreamEndpoints::learning("job-123");

        assert_eq!(endpoints.status, Some("/learning/jobs/job-123".to_string()));
        assert_eq!(
            endpoints.events,
            Some("/learning/jobs/job-123/events".to_string())
        );
        assert_eq!(
            endpoints.metrics,
            Some("/learning/jobs/job-123/metrics".to_string())
        );
        assert_eq!(
            endpoints.timeline,
            Some("/learning/jobs/job-123/timeline".to_string())
        );
    }

    #[test]
    fn test_prompt_learning_endpoints() {
        let endpoints = StreamEndpoints::prompt_learning("job-456");

        assert_eq!(
            endpoints.status,
            Some("/api/v1/offline/jobs/job-456".to_string())
        );
        assert!(endpoints.timeline.is_none());
        assert_eq!(endpoints.status_stricts.len(), 0);
    }

    #[test]
    fn test_eval_endpoints() {
        let endpoints = StreamEndpoints::eval("eval-789");

        assert_eq!(
            endpoints.status,
            Some("/api/v1/offline/jobs/eval-789".to_string())
        );
    }

    #[test]
    fn test_events_stream_url() {
        let endpoints = StreamEndpoints::learning("job-123");

        assert_eq!(
            endpoints.events_stream_url(),
            Some("/learning/jobs/job-123/events/stream".to_string())
        );
    }

    #[test]
    fn test_all_endpoints() {
        let endpoints = StreamEndpoints::prompt_learning("job-123");

        let status_endpoints = endpoints.all_status_endpoints();
        assert_eq!(status_endpoints.len(), 1);

        let event_endpoints = endpoints.all_event_endpoints();
        assert_eq!(event_endpoints.len(), 1);
    }

    #[test]
    fn test_custom_endpoints() {
        let endpoints = StreamEndpoints::custom(
            Some("/custom/status".to_string()),
            Some("/custom/events".to_string()),
            None,
            None,
        )
        .with_status_strict("/strict/status");

        assert_eq!(endpoints.status, Some("/custom/status".to_string()));
        assert_eq!(endpoints.status_stricts.len(), 1);
    }
}
