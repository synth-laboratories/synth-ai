import uuid


def get_system_id(system_name: str) -> str:
    """Create a deterministic system_instance_id from system_name using UUID5."""
    if not system_name:
        raise ValueError("system_name cannot be empty")
    system_id = uuid.uuid5(uuid.NAMESPACE_DNS, system_name)
    return str(system_id)
