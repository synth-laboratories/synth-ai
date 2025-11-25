"""Unit tests for TaskAppConfig and create_task_app."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import APIRouter, FastAPI

from synth_ai.sdk.task.contracts import TaskInfo
from synth_ai.sdk.task.server import TaskAppConfig, create_task_app


@pytest.fixture
def base_task_info() -> TaskInfo:
    """Create a base TaskInfo for testing."""
    return TaskInfo(
        task={"id": "test", "name": "Test", "version": "1.0.0"},
        environment="test",
        dataset={"id": "test", "name": "Test", "version": "1.0.0"},
        rubric={"version": "1", "criteria_count": 1, "source": "inline"},
        inference={"supports_proxy": False},
        limits={"max_turns": 10},
    )


@pytest.fixture
def minimal_config(base_task_info: TaskInfo) -> TaskAppConfig:
    """Create a minimal TaskAppConfig for testing."""
    return TaskAppConfig(
        app_id="test_app",
        name="Test App",
        description="A test task app",
        base_task_info=base_task_info,
        describe_taskset=lambda: {"id": "test", "name": "Test"},
        provide_task_instances=lambda seeds: [],
        rollout=AsyncMock(return_value=Mock()),
    )


class TestTaskAppConfig:
    """Tests for TaskAppConfig."""
    
    def test_config_creation(self, minimal_config: TaskAppConfig) -> None:
        """Test basic config creation."""
        assert minimal_config.app_id == "test_app"
        assert minimal_config.name == "Test App"
        assert minimal_config.description == "A test task app"
        assert minimal_config.require_api_key is True
        assert minimal_config.expose_debug_env is True
    
    def test_config_clone(self, minimal_config: TaskAppConfig) -> None:
        """Test that clone() creates a shallow copy."""
        cloned = minimal_config.clone()
        assert cloned.app_id == minimal_config.app_id
        assert cloned.name == minimal_config.name
        assert cloned is not minimal_config
        
        # Modifying cloned should not affect original
        cloned.app_state["new_key"] = "new_value"
        assert "new_key" not in minimal_config.app_state
    
    def test_config_defaults(self, base_task_info: TaskInfo) -> None:
        """Test that config has sensible defaults."""
        config = TaskAppConfig(
            app_id="test",
            name="Test",
            description="Test",
            base_task_info=base_task_info,
            describe_taskset=lambda: {},
            provide_task_instances=lambda seeds: [],
            rollout=AsyncMock(),
        )
        
        assert config.dataset_registry is None
        assert config.rubrics is not None
        assert config.proxy is None
        assert config.routers == ()
        assert config.middleware == ()
        assert config.app_state == {}
        assert config.require_api_key is True
        assert config.expose_debug_env is True
        assert config.cors_origins is None
        assert config.startup_hooks == ()
        assert config.shutdown_hooks == ()


class TestCreateTaskApp:
    """Tests for create_task_app function."""
    
    def test_create_app_basic(self, minimal_config: TaskAppConfig) -> None:
        """Test creating a basic FastAPI app from config."""
        app = create_task_app(minimal_config)
        
        assert isinstance(app, FastAPI)
        assert app.title == "Test App"
        assert app.description == "A test task app"
    
    def test_create_app_with_routers(self, minimal_config: TaskAppConfig) -> None:
        """Test creating app with additional routers."""
        router = APIRouter()
        
        @router.get("/custom")
        def custom_endpoint():
            return {"custom": True}
        
        minimal_config.routers = (router,)
        app = create_task_app(minimal_config)
        
        # Check that custom router is included
        routes = [route.path for route in app.routes]
        assert "/custom" in routes
    
    def test_create_app_with_cors(self, minimal_config: TaskAppConfig) -> None:
        """Test creating app with CORS middleware."""
        minimal_config.cors_origins = ["https://example.com"]
        app = create_task_app(minimal_config)
        
        # CORS middleware should be added
        assert len(app.middleware_stack.__self__.middleware) > 0
    
    def test_create_app_app_state(self, minimal_config: TaskAppConfig) -> None:
        """Test that app_state is accessible on app."""
        minimal_config.app_state["test_key"] = "test_value"
        app = create_task_app(minimal_config)
        
        assert hasattr(app.state, "test_key")
        assert app.state.test_key == "test_value"
    
    def test_create_app_health_endpoint(self, minimal_config: TaskAppConfig) -> None:
        """Test that health endpoint is created."""
        app = create_task_app(minimal_config)
        
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/" in routes  # Root endpoint
    
    def test_create_app_info_endpoint(self, minimal_config: TaskAppConfig) -> None:
        """Test that info endpoint is created."""
        app = create_task_app(minimal_config)
        
        routes = [route.path for route in app.routes]
        assert "/info" in routes
        assert "/task_info" in routes
    
    def test_create_app_rollout_endpoint(self, minimal_config: TaskAppConfig) -> None:
        """Test that rollout endpoint is created."""
        app = create_task_app(minimal_config)
        
        routes = [route.path for route in app.routes]
        assert "/rollout" in routes


