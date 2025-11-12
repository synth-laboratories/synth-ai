"""Fast unit tests for error tracking in synth-ai SDK.

These tests verify error tracking functionality that would be used
via the SDK when configuring prompt learning jobs.
"""

import pytest

# Import from backend since error tracking is backend functionality
# that gets exposed via SDK configs
from backend.app.routes.prompt_learning.core.error_tracking import (
    TerminationConditions,
    ErrorTracker,
)


class TestTerminationConditionsFast:
    """Fast unit tests for TerminationConditions."""
    
    def test_default_conditions(self):
        """Test default termination conditions."""
        conditions = TerminationConditions()
        assert conditions.max_consecutive_failures is None
        assert conditions.max_consecutive_500s is None
        assert conditions.max_error_rate is None
        assert conditions.error_rate_window_size == 100
    
    def test_custom_conditions(self):
        """Test custom termination conditions."""
        conditions = TerminationConditions(
            max_consecutive_500s=25,
            max_consecutive_failures=50,
            max_error_rate=0.5,
        )
        assert conditions.max_consecutive_500s == 25
        assert conditions.max_consecutive_failures == 50
        assert conditions.max_error_rate == 0.5
    
    def test_invalid_error_rate_raises(self):
        """Test that invalid error rate raises ValueError."""
        with pytest.raises(ValueError, match="max_error_rate must be between"):
            TerminationConditions(max_error_rate=1.5)
        
        with pytest.raises(ValueError, match="max_error_rate must be between"):
            TerminationConditions(max_error_rate=-0.1)
    
    def test_invalid_window_size_raises(self):
        """Test that invalid window size raises ValueError."""
        with pytest.raises(ValueError, match="error_rate_window_size must be > 0"):
            TerminationConditions(error_rate_window_size=0)


class TestErrorTrackerFast:
    """Fast unit tests for ErrorTracker."""
    
    def test_record_success_resets_counters(self):
        """Test that recording success resets consecutive counters."""
        conditions = TerminationConditions(max_consecutive_failures=5)
        tracker = ErrorTracker(conditions)
        
        # Record failures
        tracker.record_error("http", status_code=500)
        tracker.record_error("http", status_code=500)
        assert tracker.consecutive_failures == 2
        assert tracker.consecutive_500s == 2
        
        # Record success
        tracker.record_success()
        assert tracker.consecutive_failures == 0
        assert tracker.consecutive_500s == 0
        assert tracker.total_requests == 3
        assert tracker.total_failures == 2
    
    def test_record_500_error(self):
        """Test recording 500 error."""
        conditions = TerminationConditions()
        tracker = ErrorTracker(conditions)
        
        tracker.record_error("http", status_code=500, error_message="Internal Server Error")
        assert tracker.total_requests == 1
        assert tracker.total_failures == 1
        assert tracker.total_500s == 1
        assert tracker.consecutive_failures == 1
        assert tracker.consecutive_500s == 1
        assert tracker.last_error["status_code"] == 500
    
    def test_record_4xx_error(self):
        """Test recording 4xx error."""
        conditions = TerminationConditions()
        tracker = ErrorTracker(conditions)
        
        tracker.record_error("http", status_code=400, error_message="Bad Request")
        assert tracker.total_requests == 1
        assert tracker.total_failures == 1
        assert tracker.total_4xxs == 1
        assert tracker.consecutive_failures == 1
        assert tracker.consecutive_4xxs == 1
        assert tracker.consecutive_500s == 0
    
    def test_record_timeout(self):
        """Test recording timeout error."""
        conditions = TerminationConditions()
        tracker = ErrorTracker(conditions)
        
        tracker.record_error("timeout", error_message="Request timeout", duration=150.0)
        assert tracker.total_requests == 1
        assert tracker.total_failures == 1
        assert tracker.total_timeouts == 1
        assert tracker.consecutive_timeouts == 1
    
    def test_record_network_error(self):
        """Test recording network error."""
        conditions = TerminationConditions()
        tracker = ErrorTracker(conditions)
        
        tracker.record_error("network", error_message="Connection refused")
        assert tracker.total_requests == 1
        assert tracker.total_failures == 1
        assert tracker.consecutive_failures == 1
    
    def test_should_terminate_consecutive_failures(self):
        """Test termination on consecutive failures."""
        conditions = TerminationConditions(max_consecutive_failures=3)
        tracker = ErrorTracker(conditions)
        
        # Record 2 failures (should not terminate)
        tracker.record_error("http", status_code=500)
        tracker.record_error("http", status_code=500)
        assert not tracker.should_terminate()[0]
        
        # Record 3rd failure (should terminate)
        tracker.record_error("http", status_code=500)
        should_terminate, reason = tracker.should_terminate()
        assert should_terminate
        assert "Consecutive failures threshold" in reason
    
    def test_should_terminate_consecutive_500s(self):
        """Test termination on consecutive 500s."""
        conditions = TerminationConditions(max_consecutive_500s=2)
        tracker = ErrorTracker(conditions)
        
        # Record 1 500 (should not terminate)
        tracker.record_error("http", status_code=500)
        assert not tracker.should_terminate()[0]
        
        # Record 2nd 500 (should terminate)
        tracker.record_error("http", status_code=500)
        should_terminate, reason = tracker.should_terminate()
        assert should_terminate
        assert "Consecutive 500 errors" in reason
    
    def test_should_terminate_error_rate(self):
        """Test termination on error rate threshold."""
        conditions = TerminationConditions(max_error_rate=0.5, error_rate_window_size=10)
        tracker = ErrorTracker(conditions)
        
        # Record 5 successes and 5 failures (50% error rate, should not terminate)
        for _ in range(5):
            tracker.record_success()
        for _ in range(5):
            tracker.record_error("http", status_code=500)
        
        assert not tracker.should_terminate()[0]
        
        # Record 1 more failure (60% error rate, should terminate)
        tracker.record_error("http", status_code=500)
        should_terminate, reason = tracker.should_terminate()
        assert should_terminate
        assert "Error rate threshold" in reason
    
    def test_get_stats(self):
        """Test getting error statistics."""
        conditions = TerminationConditions()
        tracker = ErrorTracker(conditions)
        
        # Record mix of successes and failures
        for _ in range(5):
            tracker.record_success()
        for _ in range(3):
            tracker.record_error("http", status_code=500)
        tracker.record_error("http", status_code=400)
        tracker.record_error("timeout")
        
        stats = tracker.get_stats()
        assert stats["total_requests"] == 10
        assert stats["total_failures"] == 5
        assert stats["total_500s"] == 3
        assert stats["total_4xxs"] == 1
        assert stats["total_timeouts"] == 1
        assert stats["consecutive_failures"] == 5  # All 5 failures happened consecutively at the end
        assert stats["overall_error_rate"] == 0.5
        assert stats["recent_error_rate"] == pytest.approx(0.5, abs=0.01)
        assert stats["last_error"] is not None
        assert stats["last_error"]["type"] == "timeout"
    
    def test_error_rate_calculation_empty_window(self):
        """Test error rate calculation with empty window."""
        conditions = TerminationConditions(max_error_rate=0.5)
        tracker = ErrorTracker(conditions)
        
        # Should not terminate with empty window
        should_terminate, reason = tracker.should_terminate()
        assert not should_terminate
    
    def test_consecutive_counters_reset_on_different_error_types(self):
        """Test that consecutive counters reset when different error types occur."""
        conditions = TerminationConditions()
        tracker = ErrorTracker(conditions)
        
        # Record 500 errors
        tracker.record_error("http", status_code=500)
        tracker.record_error("http", status_code=500)
        assert tracker.consecutive_500s == 2
        assert tracker.consecutive_4xxs == 0
        
        # Record 4xx error (should reset 500 counter)
        tracker.record_error("http", status_code=400)
        assert tracker.consecutive_500s == 0
        assert tracker.consecutive_4xxs == 1
        
        # Record timeout (does NOT reset 4xx counter - only success or different HTTP error resets it)
        tracker.record_error("timeout")
        assert tracker.consecutive_4xxs == 1  # 4xx counter persists (timeout doesn't reset it)
        assert tracker.consecutive_timeouts == 1
    
    def test_timeout_detection_by_duration(self):
        """Test that timeouts are detected by duration threshold."""
        conditions = TerminationConditions(timeout_threshold=120.0)
        tracker = ErrorTracker(conditions)
        
        # Record error with duration > threshold
        tracker.record_error("http", status_code=500, duration=150.0)
        assert tracker.total_timeouts == 1
        assert tracker.consecutive_timeouts == 1
    
    def test_multiple_termination_conditions(self):
        """Test that any termination condition can trigger early stopping."""
        conditions = TerminationConditions(
            max_consecutive_failures=5,
            max_consecutive_500s=3,
            max_error_rate=0.8,
        )
        tracker = ErrorTracker(conditions)
        
        # Trigger consecutive 500s first (should terminate)
        tracker.record_error("http", status_code=500)
        tracker.record_error("http", status_code=500)
        tracker.record_error("http", status_code=500)
        
        should_terminate, reason = tracker.should_terminate()
        assert should_terminate
        assert "Consecutive 500 errors" in reason

