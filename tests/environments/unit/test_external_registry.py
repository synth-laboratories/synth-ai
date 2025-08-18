"""
Unit tests for external environment registry functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from synth_ai.environments.service.external_registry import (
    ExternalRegistryConfig,
    load_external_environments,
)


@pytest.mark.fast
# Include in ultra-fast unit suite
pytestmark = [pytest.mark.unit]
class TestExternalRegistry:
    """Test the external environment registry functions."""

    def test_external_registry_config_init(self):
        """Test ExternalRegistryConfig initialization."""
        # Test with no environments
        config1 = ExternalRegistryConfig()
        assert config1.external_environments == []

        # Test with environments
        envs = [{"module": "test_module", "function": "register"}]
        config2 = ExternalRegistryConfig(external_environments=envs)
        assert config2.external_environments == envs

    @patch("synth_ai.environments.service.external_registry.importlib.import_module")
    def test_load_external_environments_success(self, mock_import):
        """Test successful loading of external environments."""
        # Setup mock module with registration function
        mock_module = MagicMock()
        mock_register = MagicMock()
        mock_module.integrate_with_environments_service = mock_register
        mock_import.return_value = mock_module

        # Create config and load
        config = ExternalRegistryConfig(external_environments=[{"module": "external_env_module"}])

        load_external_environments(config)

        # Verify
        mock_import.assert_called_once_with("external_env_module")
        mock_register.assert_called_once()

    @patch("synth_ai.environments.service.external_registry.importlib.import_module")
    def test_load_external_environments_custom_function(self, mock_import):
        """Test loading with custom registration function name."""
        # Setup mock module
        mock_module = MagicMock()
        mock_custom_register = MagicMock()
        mock_module.custom_register = mock_custom_register
        mock_import.return_value = mock_module

        # Create config with custom function
        config = ExternalRegistryConfig(
            external_environments=[{"module": "external_module", "function": "custom_register"}]
        )

        load_external_environments(config)

        # Verify
        mock_import.assert_called_once_with("external_module")
        mock_custom_register.assert_called_once()

    @patch("synth_ai.environments.service.external_registry.importlib.import_module")
    @patch("synth_ai.environments.service.external_registry.logger")
    def test_load_external_environments_import_error(self, mock_logger, mock_import):
        """Test handling of import errors."""
        mock_import.side_effect = ImportError("Module not found")

        config = ExternalRegistryConfig(external_environments=[{"module": "non_existent_module"}])

        load_external_environments(config)

        # Should log error but not crash
        mock_logger.error.assert_called()

    @patch("synth_ai.environments.service.external_registry.importlib.import_module")
    @patch("synth_ai.environments.service.external_registry.logger")
    def test_load_external_environments_missing_function(self, mock_logger, mock_import):
        """Test handling when module lacks registration function."""
        # Mock module without the expected function
        mock_module = MagicMock()
        delattr(mock_module, "integrate_with_environments_service")
        mock_import.return_value = mock_module

        config = ExternalRegistryConfig(external_environments=[{"module": "incomplete_module"}])

        load_external_environments(config)

        # Should log warning
        mock_logger.warning.assert_called()

    @patch("synth_ai.environments.service.external_registry.logger")
    def test_load_external_environments_missing_module_field(self, mock_logger):
        """Test handling of config missing module field."""
        config = ExternalRegistryConfig(
            external_environments=[
                {"function": "register"}  # Missing 'module'
            ]
        )

        load_external_environments(config)

        # Should log warning
        mock_logger.warning.assert_called_with("External environment config missing 'module' field")

    @patch("synth_ai.environments.service.external_registry.importlib.import_module")
    @patch("synth_ai.environments.service.external_registry.logger")
    def test_load_external_environments_registration_error(self, mock_logger, mock_import):
        """Test handling of errors during registration."""
        # Mock module with failing registration
        mock_module = MagicMock()
        mock_module.integrate_with_environments_service.side_effect = Exception(
            "Registration failed"
        )
        mock_import.return_value = mock_module

        config = ExternalRegistryConfig(external_environments=[{"module": "failing_module"}])

        load_external_environments(config)

        # Should log error
        mock_logger.error.assert_called()
