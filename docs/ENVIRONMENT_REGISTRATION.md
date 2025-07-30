# Environment Registration Guide

This guide shows you how to register custom environments with the Synth AI environment service daemon without editing the core codebase.

## Environment Service Setup

Before registering environments, start the Synth AI service daemon:

```bash
uvx synth-ai serve
```

This command starts two services:
- **Turso Database Daemon** (port 8080) - Local SQLite database with sync capabilities
- **Environment Service API** (port 8901) - REST API for environment registration and execution

The database replicas sync every 2 seconds by default, providing local-first development with optional remote synchronization.

## Overview

There are several ways to register custom environments:

1. **Entry Points Plugin System** (Recommended) - Automatic discovery via `pyproject.toml`
2. **REST API** - Dynamic registration at runtime
3. **CLI Commands** - Command-line environment management
4. **Import-time Registration** - Manual registration in Python code

## Method 1: Entry Points Plugin System (Recommended)

This is the most Pythonic and user-friendly approach. Your environment is automatically discovered when your package is installed.

### 1.1 Create Your Environment

```python
# my_package/environments/my_env.py
from synth_ai.environments.stateful.core import StatefulEnvironment

class MyCustomEnvironment(StatefulEnvironment):
    """A custom environment for demonstration."""
    
    async def initialize(self):
        """Initialize the environment and return initial observation."""
        self.state = {"counter": 0, "done": False}
        return self.state
    
    async def step(self, tool_calls):
        """Execute a step and return new observation."""
        for call in tool_calls:
            if call.tool == "increment":
                self.state["counter"] += call.args.get("amount", 1)
            elif call.tool == "reset":
                self.state["counter"] = 0
        
        self.state["done"] = self.state["counter"] >= 10
        return self.state
    
    async def validate_tool_calls(self, tool_calls):
        """Validate tool calls."""
        valid_tools = {"increment", "reset"}
        return [call for call in tool_calls if call.tool in valid_tools]
    
    async def terminate(self):
        """Clean up and return final observation."""
        return self.state
    
    async def checkpoint(self):
        """Create a checkpoint of current state."""
        return {"state": self.state.copy()}
```

### 1.2 Register via Entry Points

Add this to your package's `pyproject.toml`:

```toml
[project.entry-points."synth_ai.environments"]
my_custom_env = "my_package.environments.my_env:MyCustomEnvironment"
another_env = "my_package.environments.other:AnotherEnvironment"
```

### 1.3 Install and Use

```bash
# Install your package
pip install my-package

# The environment is automatically available!
# Start the service
synth-ai serve

# List environments (you'll see your environment)
synth-ai env list
```

## Method 2: REST API Registration

For dynamic registration at runtime without restarting the service.

### 2.1 Register Environment

```bash
curl -X POST http://localhost:8901/registry/environments \
     -H "Content-Type: application/json" \
     -d '{
       "name": "MyCustomEnv-v1",
       "module_path": "my_package.environments.my_env",
       "class_name": "MyCustomEnvironment",
       "description": "A custom environment for testing"
     }'
```

### 2.2 List Environments

```bash
curl http://localhost:8901/registry/environments
```

### 2.3 Unregister Environment

```bash
curl -X DELETE http://localhost:8901/registry/environments/MyCustomEnv-v1
```

## Method 3: CLI Commands

Use the command-line interface for environment management.

### 3.1 Register Environment

```bash
synth-ai env register \
  --name "MyCustomEnv-v1" \
  --module "my_package.environments.my_env" \
  --class-name "MyCustomEnvironment" \
  --description "A custom environment for testing"
```

### 3.2 List Environments

```bash
synth-ai env list
```

### 3.3 Unregister Environment

```bash
synth-ai env unregister --name "MyCustomEnv-v1"
```

## Method 4: Import-time Registration

For simple cases or when you want explicit control.

### 4.1 Register in Python Code

```python
# my_package/environments/__init__.py
from synth_ai.environments.environment.registry import register_environment
from .my_env import MyCustomEnvironment

# Register when the module is imported
register_environment("MyCustomEnv-v1", MyCustomEnvironment)
```

### 4.2 Ensure Import

Make sure your module gets imported by the service. You can do this by:

1. Adding it to `PYTHONPATH`
2. Installing as an editable package: `pip install -e .`
3. Importing it in your application startup code

## Using Your Registered Environment

Once registered via any method, your environment can be used like any built-in environment:

### Via HTTP API

```bash
# Initialize environment
curl -X POST http://localhost:8901/env/MyCustomEnv-v1/initialize \
     -H "Content-Type: application/json" \
     -d '{}'

# Step environment
curl -X POST http://localhost:8901/env/MyCustomEnv-v1/step \
     -H "Content-Type: application/json" \
     -d '{
       "env_id": "your-env-id",
       "action": {
         "tool_calls": [
           {
             "tool": "increment",
             "args": {"amount": 2}
           }
         ]
       }
     }'
```

### Via Python

```python
from synth_ai.environments.environment.registry import get_environment_cls

# Get your environment class
MyEnvClass = get_environment_cls("MyCustomEnv-v1")

# Create and use instance
env = MyEnvClass(task_config)
obs = await env.initialize()
```

## Best Practices

1. **Use Entry Points**: This is the cleanest approach for distributable packages
2. **Descriptive Names**: Use versioned names like `MyEnv-v1`, `MyEnv-v2`
3. **Proper Validation**: Implement `validate_tool_calls` to ensure valid actions
4. **Documentation**: Add docstrings explaining your environment's behavior
5. **Error Handling**: Implement robust error handling in your methods
6. **Testing**: Test your environment thoroughly before deployment

## Debugging

### Check Registration Status

```bash
# List all registered environments
synth-ai env list

# Check service health
curl http://localhost:8901/health
```

### Common Issues

1. **Import Errors**: Make sure your module is on `PYTHONPATH`
2. **Class Not Found**: Check that class name matches exactly
3. **Not a StatefulEnvironment**: Ensure your class inherits from `StatefulEnvironment`
4. **Service Not Running**: Start the service with `synth-ai serve`

### Logs

Check the service logs for detailed error messages:

```bash
# If running in foreground, logs appear in terminal
synth-ai serve

# Check for import and registration messages
```

## Example Package Structure

```
my-environment-package/
├── pyproject.toml              # Entry points configuration
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── environments/
│           ├── __init__.py     # Optional registration code
│           ├── my_env.py       # Your environment class
│           └── tasks/          # Task definitions
│               └── __init__.py
└── tests/
    └── test_my_env.py         # Environment tests
```

This system provides maximum flexibility while keeping the core framework lightweight and extensible!