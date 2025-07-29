"""Abstract base class for trace storage implementations."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd

from ..session_tracer import SessionTrace


class TraceStorage(ABC):
    """Abstract base class for trace storage implementations."""
    
    @abstractmethod
    def __enter__(self):
        """Context manager entry."""
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the storage connection."""
        pass
    
    @abstractmethod
    def insert_session_trace(self, trace: SessionTrace) -> None:
        """Insert a complete session trace.
        
        Args:
            trace: The session trace to insert
        """
        pass
    
    @abstractmethod
    def query_traces(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame.
        
        Args:
            query: The query to execute
            params: Optional query parameters
            
        Returns:
            Query results as a pandas DataFrame
        """
        pass
    
    @abstractmethod
    def get_model_usage(self, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get model usage statistics.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Model usage statistics as DataFrame
        """
        pass
    
    @abstractmethod
    def get_expensive_calls(self, cost_threshold: float) -> pd.DataFrame:
        """Get expensive LLM calls above threshold.
        
        Args:
            cost_threshold: Cost threshold for filtering
            
        Returns:
            Expensive calls as DataFrame
        """
        pass
    
    @abstractmethod
    def get_session_summary(self, session_id: Optional[str] = None) -> pd.DataFrame:
        """Get session summary information.
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            Session summary as DataFrame
        """
        pass
    
    @abstractmethod
    def batch_upload(self, traces: List[SessionTrace], batch_size: int = 1000) -> None:
        """Upload multiple traces efficiently in batches.
        
        Args:
            traces: List of traces to upload
            batch_size: Size of each batch
        """
        pass
    
    # Experiment and System Management (optional methods with default implementations)
    
    def create_system(self, system_id: str, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new system. Override if supported."""
        raise NotImplementedError("System management not supported by this storage backend")
    
    def create_system_version(self, version_id: str, system_id: str, branch: str, 
                            commit: str, description: str = "") -> Dict[str, Any]:
        """Create a new system version. Override if supported."""
        raise NotImplementedError("System management not supported by this storage backend")
    
    def create_experiment(self, experiment_id: str, name: str, description: str = "",
                         system_versions: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Create a new experiment. Override if supported."""
        raise NotImplementedError("Experiment management not supported by this storage backend")
    
    def link_session_to_experiment(self, session_id: str, experiment_id: str) -> None:
        """Link a session to an experiment. Override if supported."""
        raise NotImplementedError("Experiment management not supported by this storage backend")
    
    def get_experiment_sessions(self, experiment_id: str) -> pd.DataFrame:
        """Get all sessions for an experiment. Override if supported."""
        raise NotImplementedError("Experiment management not supported by this storage backend")
    
    def get_experiments_by_system_version(self, system_version_id: str) -> pd.DataFrame:
        """Get experiments using a system version. Override if supported."""
        raise NotImplementedError("Experiment management not supported by this storage backend")
    
    def export_to_parquet(self, output_dir: str) -> None:
        """Export data to Parquet format. Override if supported."""
        raise NotImplementedError("Parquet export not supported by this storage backend")