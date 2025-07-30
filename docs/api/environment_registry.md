---
title: 'Environment Registry'
description: 'Global registry system for environment types'
---

# Environment Registry

The environment registry provides a global system for registering and retrieving environment types.

## Functions

### get_environment_cls

Retrieve a registered environment class by name.

**Signature:** `get_environment_cls(env_type: str) -> Type[StatefulEnvironment]`

**Parameters:**
- `env_type` - The name of the environment type to retrieve

**Returns:**
- The environment class that can be instantiated

**Raises:**
- `ValueError` - If env_type is not found in the registry

### list_supported_env_types

List all registered environment type names.

**Signature:** `list_supported_env_types() -> List[str]`

**Returns:**
- Sorted list of all registered environment type names

### register_environment

Register an environment class under a unique name.

**Signature:** `register_environment(env_type: str, env_cls: Type[StatefulEnvironment]) -> None`

**Parameters:**
- `env_type` - The name to register the environment under
- `env_cls` - The environment class to register

