"""Custom exceptions for storage layer."""


class StorageError(Exception):
    """Base exception for storage errors."""

    pass


class ConnectionError(StorageError):
    """Error connecting to storage backend."""

    pass


class SchemaError(StorageError):
    """Error with database schema."""

    pass


class DataValidationError(StorageError):
    """Error validating data before storage."""

    pass


class StorageNotInitializedError(StorageError):
    """Storage backend not initialized."""

    pass


class SessionNotFoundError(StorageError):
    """Session not found in storage."""

    pass


class ExperimentNotFoundError(StorageError):
    """Experiment not found in storage."""

    pass
