//! Polling and retry utilities.
//!
//! This module provides exponential backoff configuration and calculation
//! for polling loops. It's designed to be used across all SDK languages.

use std::time::Duration;

/// Configuration for exponential backoff.
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Base interval in milliseconds (default: 5000ms = 5s)
    pub base_interval_ms: u64,
    /// Maximum backoff in milliseconds (default: 60000ms = 60s)
    pub max_backoff_ms: u64,
    /// Maximum exponent for backoff calculation (default: 4, giving max multiplier of 16)
    pub max_exponent: u32,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            base_interval_ms: 5000, // 5 seconds
            max_backoff_ms: 60000,  // 60 seconds
            max_exponent: 4,        // 2^4 = 16x max multiplier
        }
    }
}

impl BackoffConfig {
    /// Create a new backoff config with custom values.
    pub fn new(base_interval_ms: u64, max_backoff_ms: u64, max_exponent: u32) -> Self {
        Self {
            base_interval_ms,
            max_backoff_ms,
            max_exponent,
        }
    }

    /// Create a fast backoff config for quick retries.
    pub fn fast() -> Self {
        Self {
            base_interval_ms: 1000, // 1 second
            max_backoff_ms: 10000,  // 10 seconds
            max_exponent: 3,        // 2^3 = 8x max multiplier
        }
    }

    /// Create an aggressive backoff config for rate-limited scenarios.
    pub fn aggressive() -> Self {
        Self {
            base_interval_ms: 10000, // 10 seconds
            max_backoff_ms: 300000,  // 5 minutes
            max_exponent: 5,         // 2^5 = 32x max multiplier
        }
    }
}

/// Calculate backoff delay for a given number of consecutive failures.
///
/// Formula: `min(base * 2^min(consecutive-1, max_exponent), max_backoff)`
///
/// # Arguments
///
/// * `config` - Backoff configuration
/// * `consecutive_failures` - Number of consecutive failures (0 = first attempt)
///
/// # Returns
///
/// Duration to wait before next attempt
///
/// # Example
///
/// ```
/// use synth_ai_core::polling::{BackoffConfig, calculate_backoff};
///
/// let config = BackoffConfig::default();
///
/// // First failure: 5s (base)
/// let delay1 = calculate_backoff(&config, 1);
/// assert_eq!(delay1.as_millis(), 5000);
///
/// // Second failure: 10s (base * 2)
/// let delay2 = calculate_backoff(&config, 2);
/// assert_eq!(delay2.as_millis(), 10000);
///
/// // Third failure: 20s (base * 4)
/// let delay3 = calculate_backoff(&config, 3);
/// assert_eq!(delay3.as_millis(), 20000);
/// ```
pub fn calculate_backoff(config: &BackoffConfig, consecutive_failures: u32) -> Duration {
    if consecutive_failures == 0 {
        return Duration::from_millis(config.base_interval_ms);
    }

    // Exponent is (consecutive - 1), capped at max_exponent
    let exponent = (consecutive_failures.saturating_sub(1)).min(config.max_exponent);
    let multiplier = 2u64.saturating_pow(exponent);
    let delay_ms = config
        .base_interval_ms
        .saturating_mul(multiplier)
        .min(config.max_backoff_ms);

    Duration::from_millis(delay_ms)
}

/// Calculate backoff delay in milliseconds (convenience function).
pub fn calculate_backoff_ms(
    base_interval_ms: u64,
    max_backoff_ms: u64,
    consecutive_failures: u32,
) -> u64 {
    let config = BackoffConfig {
        base_interval_ms,
        max_backoff_ms,
        max_exponent: 4, // default
    };
    calculate_backoff(&config, consecutive_failures).as_millis() as u64
}

/// Result of a single poll operation.
#[derive(Debug, Clone)]
pub enum PollResult<T> {
    /// Continue polling (no terminal state reached)
    Continue,
    /// Polling complete with terminal result
    Terminal(T),
    /// Polling encountered an error (may retry)
    Error(String),
}

/// Configuration for a polling loop.
#[derive(Debug, Clone)]
pub struct PollConfig {
    /// Backoff configuration for failures
    pub backoff: BackoffConfig,
    /// Maximum number of consecutive errors before giving up
    pub max_consecutive_errors: u32,
    /// Overall timeout in seconds (0 = no timeout)
    pub timeout_secs: u64,
    /// Base polling interval when no errors (milliseconds)
    pub poll_interval_ms: u64,
}

impl Default for PollConfig {
    fn default() -> Self {
        Self {
            backoff: BackoffConfig::default(),
            max_consecutive_errors: 5,
            timeout_secs: 0,        // no timeout
            poll_interval_ms: 5000, // 5 seconds
        }
    }
}

/// State tracker for a polling loop.
#[derive(Debug)]
pub struct PollState {
    /// Number of consecutive errors
    pub consecutive_errors: u32,
    /// Total number of poll attempts
    pub total_attempts: u32,
    /// Start time (for timeout tracking)
    start_time: std::time::Instant,
    /// Configuration
    config: PollConfig,
}

impl PollState {
    /// Create a new polling state.
    pub fn new(config: PollConfig) -> Self {
        Self {
            consecutive_errors: 0,
            total_attempts: 0,
            start_time: std::time::Instant::now(),
            config,
        }
    }

    /// Record a successful poll (resets consecutive errors).
    pub fn record_success(&mut self) {
        self.consecutive_errors = 0;
        self.total_attempts += 1;
    }

    /// Record a failed poll.
    pub fn record_error(&mut self) {
        self.consecutive_errors += 1;
        self.total_attempts += 1;
    }

    /// Check if we should give up due to too many errors.
    pub fn should_give_up(&self) -> bool {
        self.consecutive_errors >= self.config.max_consecutive_errors
    }

    /// Check if we've timed out.
    pub fn is_timed_out(&self) -> bool {
        if self.config.timeout_secs == 0 {
            return false;
        }
        self.start_time.elapsed().as_secs() >= self.config.timeout_secs
    }

    /// Get the next delay to wait.
    pub fn next_delay(&self) -> Duration {
        if self.consecutive_errors > 0 {
            calculate_backoff(&self.config.backoff, self.consecutive_errors)
        } else {
            Duration::from_millis(self.config.poll_interval_ms)
        }
    }

    /// Get elapsed time since polling started.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Retry helper for async operations.
///
/// # Example
///
/// ```ignore
/// use synth_ai_core::polling::RetryConfig;
///
/// let config = RetryConfig::default();
/// let result = config.retry(|| async {
///     // Your fallible operation here
///     fetch_data().await
/// }).await?;
/// ```
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of attempts (including first try)
    pub max_attempts: u32,
    /// Initial delay between retries in milliseconds
    pub initial_delay_ms: u64,
    /// Backoff multiplier (e.g., 2.0 for doubling)
    pub backoff_multiplier: f64,
    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30000,
        }
    }
}

impl RetryConfig {
    /// Calculate delay for a given attempt number (0-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(self.initial_delay_ms);
        }

        let multiplier = self.backoff_multiplier.powi(attempt as i32);
        let delay_ms = (self.initial_delay_ms as f64 * multiplier) as u64;
        Duration::from_millis(delay_ms.min(self.max_delay_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_backoff_default() {
        let config = BackoffConfig::default();

        // First failure: base (5s)
        assert_eq!(calculate_backoff(&config, 1).as_millis(), 5000);

        // Second failure: base * 2 (10s)
        assert_eq!(calculate_backoff(&config, 2).as_millis(), 10000);

        // Third failure: base * 4 (20s)
        assert_eq!(calculate_backoff(&config, 3).as_millis(), 20000);

        // Fourth failure: base * 8 (40s)
        assert_eq!(calculate_backoff(&config, 4).as_millis(), 40000);

        // Fifth failure: base * 16 = 80s, but capped at 60s
        assert_eq!(calculate_backoff(&config, 5).as_millis(), 60000);

        // Sixth+ failure: still capped at 60s
        assert_eq!(calculate_backoff(&config, 6).as_millis(), 60000);
        assert_eq!(calculate_backoff(&config, 10).as_millis(), 60000);
    }

    #[test]
    fn test_calculate_backoff_zero_failures() {
        let config = BackoffConfig::default();
        assert_eq!(calculate_backoff(&config, 0).as_millis(), 5000);
    }

    #[test]
    fn test_calculate_backoff_fast() {
        let config = BackoffConfig::fast();
        // Fast config: base=1000, max=10000, max_exponent=3
        assert_eq!(calculate_backoff(&config, 1).as_millis(), 1000); // 1000 * 2^0 = 1000
        assert_eq!(calculate_backoff(&config, 2).as_millis(), 2000); // 1000 * 2^1 = 2000
        assert_eq!(calculate_backoff(&config, 3).as_millis(), 4000); // 1000 * 2^2 = 4000
        assert_eq!(calculate_backoff(&config, 4).as_millis(), 8000); // 1000 * 2^3 = 8000
                                                                     // max_exponent=3, so exponent is capped at 3 (8x multiplier)
        assert_eq!(calculate_backoff(&config, 5).as_millis(), 8000); // 1000 * 2^3 = 8000 (capped)
        assert_eq!(calculate_backoff(&config, 10).as_millis(), 8000); // 1000 * 2^3 = 8000 (capped)
    }

    #[test]
    fn test_poll_state() {
        let config = PollConfig {
            max_consecutive_errors: 3,
            timeout_secs: 0,
            ..Default::default()
        };

        let mut state = PollState::new(config);

        // Initially no errors
        assert!(!state.should_give_up());
        assert_eq!(state.consecutive_errors, 0);

        // Record some errors
        state.record_error();
        assert!(!state.should_give_up());
        state.record_error();
        assert!(!state.should_give_up());
        state.record_error();
        assert!(state.should_give_up());

        // Success resets
        state.record_success();
        assert!(!state.should_give_up());
        assert_eq!(state.consecutive_errors, 0);
    }

    #[test]
    fn test_retry_delay() {
        let config = RetryConfig {
            max_attempts: 5,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 10000,
        };

        assert_eq!(config.delay_for_attempt(0).as_millis(), 1000);
        assert_eq!(config.delay_for_attempt(1).as_millis(), 2000);
        assert_eq!(config.delay_for_attempt(2).as_millis(), 4000);
        assert_eq!(config.delay_for_attempt(3).as_millis(), 8000);
        // Capped at 10s
        assert_eq!(config.delay_for_attempt(4).as_millis(), 10000);
    }
}
