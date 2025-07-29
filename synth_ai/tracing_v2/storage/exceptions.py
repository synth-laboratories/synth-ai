"""Custom exceptions for trace storage."""


class TraceStorageError(Exception):
    """Base exception for trace storage errors."""
    pass


class SessionNotFoundError(TraceStorageError):
    """Raised when a session is not found."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionAlreadyExistsError(TraceStorageError):
    """Raised when trying to insert a duplicate session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session already exists: {session_id}")


class DatabaseConnectionError(TraceStorageError):
    """Raised when database connection fails."""
    pass


class SchemaInitializationError(TraceStorageError):
    """Raised when schema initialization fails."""
    pass


class QueryExecutionError(TraceStorageError):
    """Raised when query execution fails."""
    def __init__(self, query: str, error: Exception):
        self.query = query
        self.original_error = error
        super().__init__(f"Query execution failed: {error}")


class DataValidationError(TraceStorageError):
    """Raised when data validation fails."""
    pass