"""
Tests for v2 tracing configuration system.

This module tests:
- TracingConfig loading and validation
- Environment variable handling
- Configuration precedence
- Dynamic configuration updates
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from synth_ai.tracing_v2.config import TracingConfig, get_config, _config_cache


class TestTracingConfigBasics:
    """Test basic TracingConfig functionality."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = TracingConfig()
        
        # Basic settings
        assert config.enabled is True
        assert config.emit_events is True
        assert config.emit_messages is True
        
        # OTel settings
        assert config.otel_enabled is False
        assert config.otel_service_name == "synth-ai"
        assert config.otel_service_version == "0.1.0"
        assert config.otel_deployment_environment == "production"
        
        # Sampling
        assert config.sampling_rate == 1.0
        assert config.otel_sampling_rate == 1.0
        
        # Batch processing
        assert config.otel_batch_max_queue_size == 2048
        assert config.otel_batch_max_export_size == 512
        assert config.otel_batch_schedule_delay_ms == 5000
        
        # Performance
        assert config.max_message_size == 1024 * 1024  # 1MB
        assert config.max_attribute_length == 10000
        assert config.payload_truncation_enabled is True
        
        # Cost tracking
        assert config.track_costs is True
        
        # PII
        assert config.mask_pii is True
        assert config.pii_patterns is not None
        assert len(config.pii_patterns) > 0
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "enabled": False,
            "otel_enabled": True,
            "otel_service_name": "test-service",
            "sampling_rate": 0.5,
            "track_costs": False
        }
        
        config = TracingConfig(**config_dict)
        
        assert config.enabled is False
        assert config.otel_enabled is True
        assert config.otel_service_name == "test-service"
        assert config.sampling_rate == 0.5
        assert config.track_costs is False
        
        # Other values should be defaults
        assert config.emit_events is True
        assert config.mask_pii is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid sampling rate
        with pytest.raises(ValueError):
            TracingConfig(sampling_rate=1.5)
        
        with pytest.raises(ValueError):
            TracingConfig(sampling_rate=-0.1)
        
        # Invalid OTel sampling rate
        with pytest.raises(ValueError):
            TracingConfig(otel_sampling_rate=2.0)
        
        # Invalid batch size
        with pytest.raises(ValueError):
            TracingConfig(otel_batch_max_export_size=0)
        
        # Invalid max message size
        with pytest.raises(ValueError):
            TracingConfig(max_message_size=-1)


class TestConfigEnvironmentVariables:
    """Test environment variable handling."""
    
    def test_env_var_override(self):
        """Test that environment variables override defaults."""
        env_vars = {
            "SYNTH_TRACING_ENABLED": "false",
            "SYNTH_TRACING_OTEL_ENABLED": "true",
            "SYNTH_TRACING_OTEL_SERVICE_NAME": "env-service",
            "SYNTH_TRACING_SAMPLING_RATE": "0.25",
            "SYNTH_TRACING_TRACK_COSTS": "false",
            "SYNTH_TRACING_MASK_PII": "false"
        }
        
        with patch.dict(os.environ, env_vars):
            config = TracingConfig.from_env()
        
        assert config.enabled is False
        assert config.otel_enabled is True
        assert config.otel_service_name == "env-service"
        assert config.sampling_rate == 0.25
        assert config.track_costs is False
        assert config.mask_pii is False
    
    def test_env_var_type_conversion(self):
        """Test proper type conversion from environment variables."""
        env_vars = {
            "SYNTH_TRACING_OTEL_BATCH_MAX_QUEUE_SIZE": "1000",
            "SYNTH_TRACING_OTEL_BATCH_SCHEDULE_DELAY_MS": "2500",
            "SYNTH_TRACING_MAX_MESSAGE_SIZE": "2097152",  # 2MB
            "SYNTH_TRACING_MAX_ATTRIBUTE_LENGTH": "5000"
        }
        
        with patch.dict(os.environ, env_vars):
            config = TracingConfig.from_env()
        
        assert config.otel_batch_max_queue_size == 1000
        assert config.otel_batch_schedule_delay_ms == 2500
        assert config.max_message_size == 2097152
        assert config.max_attribute_length == 5000
    
    def test_env_var_boolean_parsing(self):
        """Test boolean parsing from environment variables."""
        # Test various true values
        for true_value in ["true", "True", "TRUE", "1", "yes", "YES"]:
            with patch.dict(os.environ, {"SYNTH_TRACING_ENABLED": true_value}):
                config = TracingConfig.from_env()
                assert config.enabled is True
        
        # Test various false values
        for false_value in ["false", "False", "FALSE", "0", "no", "NO"]:
            with patch.dict(os.environ, {"SYNTH_TRACING_ENABLED": false_value}):
                config = TracingConfig.from_env()
                assert config.enabled is False
    
    def test_env_var_invalid_values(self):
        """Test handling of invalid environment variable values."""
        # Invalid boolean
        with patch.dict(os.environ, {"SYNTH_TRACING_ENABLED": "maybe"}):
            config = TracingConfig.from_env()
            assert config.enabled is True  # Should use default
        
        # Invalid float
        with patch.dict(os.environ, {"SYNTH_TRACING_SAMPLING_RATE": "not-a-number"}):
            config = TracingConfig.from_env()
            assert config.sampling_rate == 1.0  # Should use default
        
        # Invalid integer
        with patch.dict(os.environ, {"SYNTH_TRACING_MAX_MESSAGE_SIZE": "huge"}):
            config = TracingConfig.from_env()
            assert config.max_message_size == 1024 * 1024  # Should use default


class TestConfigFileLoading:
    """Test configuration file loading."""
    
    def test_load_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_data = {
            "enabled": True,
            "otel_enabled": True,
            "otel_service_name": "json-service",
            "sampling_rate": 0.75,
            "pii_patterns": {
                "custom_pattern": r"\b[A-Z]{3}-\d{4}\b"
            }
        }
        
        config_file = tmp_path / "tracing_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        config = TracingConfig.from_file(str(config_file))
        
        assert config.enabled is True
        assert config.otel_enabled is True
        assert config.otel_service_name == "json-service"
        assert config.sampling_rate == 0.75
        assert "custom_pattern" in config.pii_patterns
    
    def test_load_from_yaml_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        pytest.importorskip("yaml")  # Skip if PyYAML not installed
        
        config_yaml = """
enabled: true
otel_enabled: true
otel_service_name: yaml-service
otel_deployment_environment: staging
sampling_rate: 0.9
track_costs: false
pii_patterns:
  account_number: "\\\\bACCT-\\\\d{8}\\\\b"
"""
        
        config_file = tmp_path / "tracing_config.yaml"
        with open(config_file, "w") as f:
            f.write(config_yaml)
        
        config = TracingConfig.from_file(str(config_file))
        
        assert config.enabled is True
        assert config.otel_enabled is True
        assert config.otel_service_name == "yaml-service"
        assert config.otel_deployment_environment == "staging"
        assert config.sampling_rate == 0.9
        assert config.track_costs is False
        assert "account_number" in config.pii_patterns
    
    def test_file_not_found(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            TracingConfig.from_file("/non/existent/file.json")
    
    def test_invalid_json_file(self, tmp_path):
        """Test handling of invalid JSON file."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, "w") as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            TracingConfig.from_file(str(config_file))


class TestConfigPrecedence:
    """Test configuration precedence and merging."""
    
    def test_precedence_order(self, tmp_path):
        """Test that precedence is: args > env > file > defaults."""
        # Create config file
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump({
                "enabled": False,
                "otel_service_name": "file-service",
                "sampling_rate": 0.5,
                "track_costs": False
            }, f)
        
        # Set environment variables
        env_vars = {
            "SYNTH_TRACING_OTEL_SERVICE_NAME": "env-service",
            "SYNTH_TRACING_SAMPLING_RATE": "0.75"
        }
        
        with patch.dict(os.environ, env_vars):
            # Load with all sources
            config = TracingConfig.load(
                config_file=str(config_file),
                otel_service_name="arg-service"  # Direct argument
            )
        
        # Check precedence
        assert config.enabled is False  # From file (no env override)
        assert config.otel_service_name == "arg-service"  # Argument wins
        assert config.sampling_rate == 0.75  # Env overrides file
        assert config.track_costs is False  # From file (no env override)
    
    def test_partial_override(self):
        """Test that only specified values are overridden."""
        base_config = TracingConfig(
            enabled=True,
            otel_enabled=False,
            otel_service_name="base-service",
            sampling_rate=1.0
        )
        
        # Override only some values
        env_vars = {
            "SYNTH_TRACING_OTEL_ENABLED": "true",
            "SYNTH_TRACING_SAMPLING_RATE": "0.5"
        }
        
        with patch.dict(os.environ, env_vars):
            config = TracingConfig.load()
        
        assert config.enabled is True  # Not overridden
        assert config.otel_enabled is True  # Overridden by env
        assert config.otel_service_name == "synth-ai"  # Default (not overridden)
        assert config.sampling_rate == 0.5  # Overridden by env


class TestConfigCaching:
    """Test configuration caching behavior."""
    
    def setUp(self):
        """Clear config cache before each test."""
        _config_cache.clear()
    
    def test_get_config_caching(self):
        """Test that get_config() caches configuration."""
        self.setUp()
        
        # First call should create config
        config1 = get_config()
        assert config1 is not None
        
        # Second call should return same instance
        config2 = get_config()
        assert config2 is config1
        
        # Verify it's the same object
        config1.otel_service_name = "modified-service"
        assert config2.otel_service_name == "modified-service"
    
    def test_config_cache_with_env_change(self):
        """Test that cache is invalidated when env vars change."""
        self.setUp()
        
        # Get initial config
        config1 = get_config()
        original_name = config1.otel_service_name
        
        # Change environment variable
        with patch.dict(os.environ, {"SYNTH_TRACING_OTEL_SERVICE_NAME": "new-service"}):
            # Clear cache to simulate app restart
            _config_cache.clear()
            
            # Get new config
            config2 = get_config()
            assert config2.otel_service_name == "new-service"
            assert config2.otel_service_name != original_name
    
    def test_config_file_path_env(self, tmp_path):
        """Test loading config from file specified in env var."""
        self.setUp()
        
        # Create config file
        config_file = tmp_path / "env_config.json"
        with open(config_file, "w") as f:
            json.dump({"otel_service_name": "env-file-service"}, f)
        
        with patch.dict(os.environ, {"SYNTH_TRACING_CONFIG_FILE": str(config_file)}):
            _config_cache.clear()
            config = get_config()
            assert config.otel_service_name == "env-file-service"


class TestPIIPatternConfiguration:
    """Test PII pattern configuration."""
    
    def test_default_pii_patterns(self):
        """Test that default PII patterns are included."""
        config = TracingConfig()
        
        assert "email" in config.pii_patterns
        assert "phone" in config.pii_patterns
        assert "ssn" in config.pii_patterns
        assert "credit_card" in config.pii_patterns
        
        # Test patterns work
        email_pattern = config.pii_patterns["email"]
        assert email_pattern.search("test@example.com") is not None
    
    def test_custom_pii_patterns(self):
        """Test adding custom PII patterns."""
        custom_patterns = {
            "employee_id": r"\bEMP-\d{6}\b",
            "account_number": r"\bACCT-[A-Z]{2}\d{8}\b"
        }
        
        config = TracingConfig(pii_patterns=custom_patterns)
        
        # Custom patterns should be added
        assert "employee_id" in config.pii_patterns
        assert "account_number" in config.pii_patterns
        
        # Default patterns should still exist
        assert "email" in config.pii_patterns
        assert "phone" in config.pii_patterns
    
    def test_override_default_pattern(self):
        """Test overriding a default PII pattern."""
        # More strict email pattern
        custom_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@company\.com\b"
        }
        
        config = TracingConfig(pii_patterns=custom_patterns)
        
        # Should match company emails
        pattern = config.pii_patterns["email"]
        assert pattern.search("user@company.com") is not None
        assert pattern.search("user@other.com") is None
    
    def test_disable_pii_masking(self):
        """Test disabling PII masking."""
        config = TracingConfig(mask_pii=False)
        
        assert config.mask_pii is False
        # Patterns should still be loaded (for potential use)
        assert len(config.pii_patterns) > 0


class TestConfigValidation:
    """Test configuration validation rules."""
    
    def test_sampling_rate_validation(self):
        """Test sampling rate validation."""
        # Valid rates
        for rate in [0.0, 0.1, 0.5, 0.99, 1.0]:
            config = TracingConfig(sampling_rate=rate)
            assert config.sampling_rate == rate
        
        # Invalid rates
        for rate in [-0.1, 1.1, 2.0, -1.0]:
            with pytest.raises(ValueError, match="sampling_rate must be between 0.0 and 1.0"):
                TracingConfig(sampling_rate=rate)
    
    def test_batch_size_validation(self):
        """Test batch processing size validation."""
        # Valid sizes
        config = TracingConfig(
            otel_batch_max_queue_size=100,
            otel_batch_max_export_size=50
        )
        assert config.otel_batch_max_queue_size == 100
        assert config.otel_batch_max_export_size == 50
        
        # Invalid sizes
        with pytest.raises(ValueError, match="must be positive"):
            TracingConfig(otel_batch_max_queue_size=0)
        
        with pytest.raises(ValueError, match="must be positive"):
            TracingConfig(otel_batch_max_export_size=-1)
    
    def test_message_size_validation(self):
        """Test message size limit validation."""
        # Valid sizes
        config = TracingConfig(
            max_message_size=1024,
            max_attribute_length=100
        )
        assert config.max_message_size == 1024
        assert config.max_attribute_length == 100
        
        # Invalid sizes
        with pytest.raises(ValueError, match="must be positive"):
            TracingConfig(max_message_size=0)
        
        with pytest.raises(ValueError, match="must be positive"):
            TracingConfig(max_attribute_length=-100)
    
    def test_service_name_validation(self):
        """Test service name validation."""
        # Valid names
        valid_names = [
            "my-service",
            "service_123",
            "Service.Name",
            "service-with-version-1.0"
        ]
        
        for name in valid_names:
            config = TracingConfig(otel_service_name=name)
            assert config.otel_service_name == name
        
        # Empty name should raise error
        with pytest.raises(ValueError, match="service_name cannot be empty"):
            TracingConfig(otel_service_name="")
        
        with pytest.raises(ValueError, match="service_name cannot be empty"):
            TracingConfig(otel_service_name="   ")