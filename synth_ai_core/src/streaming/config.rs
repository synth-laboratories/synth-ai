//! Stream configuration for filtering and controlling stream behavior.

use super::types::StreamType;
use serde_json::Value;
use std::collections::HashSet;

/// Configuration for stream filtering and behavior.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Which stream types to enable.
    pub enabled_streams: HashSet<StreamType>,
    /// Whitelist of event types to include (None = include all).
    pub event_types: Option<HashSet<String>>,
    /// Blacklist of event types to exclude.
    pub event_types_exclude: Option<HashSet<String>>,
    /// Filter by event levels (e.g., "error", "warning", "info").
    pub event_levels: Option<HashSet<String>>,
    /// Filter metrics by name.
    pub metric_names: Option<HashSet<String>>,
    /// Filter metrics by phase.
    pub metric_phases: Option<HashSet<String>>,
    /// Filter timeline entries by phase.
    pub timeline_phases: Option<HashSet<String>>,
    /// Sampling rate (0.0-1.0) for events.
    pub sample_rate: f64,
    /// Maximum events to return per poll.
    pub max_events_per_poll: Option<usize>,
    /// Enable deduplication.
    pub deduplicate: bool,
    /// Polling interval in seconds.
    pub poll_interval_seconds: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self::with_default_filters()
    }
}

impl StreamConfig {
    /// Create a config that includes all streams with no filtering.
    pub fn all() -> Self {
        Self {
            enabled_streams: [
                StreamType::Status,
                StreamType::Events,
                StreamType::Metrics,
                StreamType::Timeline,
            ]
            .into_iter()
            .collect(),
            event_types: None,
            event_types_exclude: None,
            event_levels: None,
            metric_names: None,
            metric_phases: None,
            timeline_phases: None,
            sample_rate: 1.0,
            max_events_per_poll: None,
            deduplicate: true,
            poll_interval_seconds: 2.0,
        }
    }

    /// Create a config with sensible default filters.
    pub fn with_default_filters() -> Self {
        let mut config = Self::all();
        // Exclude noisy internal events by default
        config.event_types_exclude = Some(
            [
                "sft.progress",
                "sft.loss",
                "sft.upstream.status",
                "internal.heartbeat",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        config
    }

    /// Create a minimal config (status only).
    pub fn minimal() -> Self {
        Self {
            enabled_streams: [StreamType::Status].into_iter().collect(),
            event_types: None,
            event_types_exclude: None,
            event_levels: None,
            metric_names: None,
            metric_phases: None,
            timeline_phases: None,
            sample_rate: 1.0,
            max_events_per_poll: None,
            deduplicate: true,
            poll_interval_seconds: 5.0,
        }
    }

    /// Create a config for errors and warnings only.
    pub fn errors_only() -> Self {
        Self {
            enabled_streams: [StreamType::Status, StreamType::Events]
                .into_iter()
                .collect(),
            event_types: None,
            event_types_exclude: None,
            event_levels: Some(["error", "warning"].iter().map(|s| s.to_string()).collect()),
            metric_names: None,
            metric_phases: None,
            timeline_phases: None,
            sample_rate: 1.0,
            max_events_per_poll: None,
            deduplicate: true,
            poll_interval_seconds: 2.0,
        }
    }

    /// Create a config for metrics only.
    pub fn metrics_only() -> Self {
        Self {
            enabled_streams: [StreamType::Status, StreamType::Metrics]
                .into_iter()
                .collect(),
            event_types: None,
            event_types_exclude: None,
            event_levels: None,
            metric_names: None,
            metric_phases: None,
            timeline_phases: None,
            sample_rate: 1.0,
            max_events_per_poll: None,
            deduplicate: true,
            poll_interval_seconds: 1.0,
        }
    }

    /// Enable a specific stream type.
    pub fn enable_stream(mut self, stream_type: StreamType) -> Self {
        self.enabled_streams.insert(stream_type);
        self
    }

    /// Disable a specific stream type.
    pub fn disable_stream(mut self, stream_type: StreamType) -> Self {
        self.enabled_streams.remove(&stream_type);
        self
    }

    /// Add an event type to the whitelist.
    pub fn include_event_type(mut self, event_type: impl Into<String>) -> Self {
        let types = self.event_types.get_or_insert_with(HashSet::new);
        types.insert(event_type.into());
        self
    }

    /// Add an event type to the blacklist.
    pub fn exclude_event_type(mut self, event_type: impl Into<String>) -> Self {
        let types = self.event_types_exclude.get_or_insert_with(HashSet::new);
        types.insert(event_type.into());
        self
    }

    /// Filter by event levels.
    pub fn with_levels(mut self, levels: Vec<&str>) -> Self {
        self.event_levels = Some(levels.into_iter().map(String::from).collect());
        self
    }

    /// Filter metrics by phase.
    pub fn with_metric_phases(mut self, phases: Vec<&str>) -> Self {
        self.metric_phases = Some(phases.into_iter().map(String::from).collect());
        self
    }

    /// Filter timeline entries by phase.
    pub fn with_timeline_phases(mut self, phases: Vec<&str>) -> Self {
        self.timeline_phases = Some(phases.into_iter().map(String::from).collect());
        self
    }

    /// Set the polling interval.
    pub fn with_interval(mut self, seconds: f64) -> Self {
        self.poll_interval_seconds = seconds;
        self
    }

    /// Set the sample rate.
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Disable deduplication.
    pub fn without_deduplication(mut self) -> Self {
        self.deduplicate = false;
        self
    }

    /// Check if a stream type is enabled.
    pub fn is_stream_enabled(&self, stream_type: StreamType) -> bool {
        self.enabled_streams.contains(&stream_type)
    }

    /// Check if an event should be included based on filters.
    pub fn should_include_event(&self, event: &Value) -> bool {
        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

        // Blacklist takes precedence
        if let Some(ref exclude) = self.event_types_exclude {
            if exclude.contains(event_type) {
                return false;
            }
        }

        // Whitelist check (if whitelist exists, event must be in it)
        if let Some(ref include) = self.event_types {
            if !include.contains(event_type) {
                return false;
            }
        }

        // Level check
        if let Some(ref levels) = self.event_levels {
            let level = event.get("level").and_then(|v| v.as_str()).unwrap_or("");
            if !levels.contains(level) {
                return false;
            }
        }

        // Sample rate check
        if self.sample_rate < 1.0 {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            event.to_string().hash(&mut hasher);
            let hash = hasher.finish();
            let threshold = (self.sample_rate * u64::MAX as f64) as u64;
            if hash > threshold {
                return false;
            }
        }

        true
    }

    /// Check if a metric should be included based on filters.
    pub fn should_include_metric(&self, metric: &Value) -> bool {
        if let Some(ref names) = self.metric_names {
            let name = metric.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if !names.contains(name) {
                return false;
            }
        }

        if let Some(ref phases) = self.metric_phases {
            let phase = metric.get("phase").and_then(|v| v.as_str()).unwrap_or("");
            return phases.contains(phase);
        }
        true
    }

    /// Check if a timeline entry should be included based on filters.
    pub fn should_include_timeline(&self, entry: &Value) -> bool {
        if let Some(ref phases) = self.timeline_phases {
            let phase = entry.get("phase").and_then(|v| v.as_str()).unwrap_or("");
            return phases.contains(phase);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_config() {
        let config = StreamConfig::all();
        assert!(config.is_stream_enabled(StreamType::Status));
        assert!(config.is_stream_enabled(StreamType::Events));
        assert!(config.is_stream_enabled(StreamType::Metrics));
        assert!(config.is_stream_enabled(StreamType::Timeline));
    }

    #[test]
    fn test_minimal_config() {
        let config = StreamConfig::minimal();
        assert!(config.is_stream_enabled(StreamType::Status));
        assert!(!config.is_stream_enabled(StreamType::Events));
    }

    #[test]
    fn test_event_blacklist() {
        let config = StreamConfig::all().exclude_event_type("internal.heartbeat");

        let allowed = serde_json::json!({"type": "progress"});
        let blocked = serde_json::json!({"type": "internal.heartbeat"});

        assert!(config.should_include_event(&allowed));
        assert!(!config.should_include_event(&blocked));
    }

    #[test]
    fn test_event_whitelist() {
        let config = StreamConfig::all()
            .include_event_type("error")
            .include_event_type("warning");

        let allowed = serde_json::json!({"type": "error"});
        let blocked = serde_json::json!({"type": "progress"});

        assert!(config.should_include_event(&allowed));
        assert!(!config.should_include_event(&blocked));
    }

    #[test]
    fn test_level_filter() {
        let config = StreamConfig::errors_only();

        let error = serde_json::json!({"type": "test", "level": "error"});
        let warning = serde_json::json!({"type": "test", "level": "warning"});
        let info = serde_json::json!({"type": "test", "level": "info"});

        assert!(config.should_include_event(&error));
        assert!(config.should_include_event(&warning));
        assert!(!config.should_include_event(&info));
    }

    #[test]
    fn test_stream_enable_disable() {
        let config = StreamConfig::all()
            .disable_stream(StreamType::Timeline)
            .disable_stream(StreamType::Metrics);

        assert!(config.is_stream_enabled(StreamType::Status));
        assert!(config.is_stream_enabled(StreamType::Events));
        assert!(!config.is_stream_enabled(StreamType::Metrics));
        assert!(!config.is_stream_enabled(StreamType::Timeline));
    }

    #[test]
    fn test_builder_pattern() {
        let config = StreamConfig::minimal()
            .enable_stream(StreamType::Events)
            .exclude_event_type("heartbeat")
            .with_interval(1.0)
            .without_deduplication();

        assert!(config.is_stream_enabled(StreamType::Events));
        assert_eq!(config.poll_interval_seconds, 1.0);
        assert!(!config.deduplicate);
    }
}
