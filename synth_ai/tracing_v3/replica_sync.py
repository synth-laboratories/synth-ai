"""Turso embedded replica synchronization."""
import asyncio
import libsql
import os
import logging
from typing import Optional
from .config import CONFIG

logger = logging.getLogger(__name__)


class ReplicaSync:
    """Manages synchronization of embedded SQLite replica with remote Turso database."""
    
    def __init__(self, 
                 db_path: str = "embedded.db",
                 sync_url: Optional[str] = None,
                 auth_token: Optional[str] = None,
                 sync_interval: Optional[int] = None):
        """Initialize replica sync manager.
        
        Args:
            db_path: Path to local embedded database file
            sync_url: Remote Turso database URL (defaults to CONFIG.sync_url)
            auth_token: Authentication token (defaults to CONFIG.auth_token) 
            sync_interval: Sync interval in seconds (defaults to CONFIG.sync_interval)
        """
        self.db_path = db_path
        self.sync_url = sync_url or CONFIG.sync_url
        self.auth_token = auth_token or CONFIG.auth_token
        self.sync_interval = sync_interval or CONFIG.sync_interval
        self._sync_task: Optional[asyncio.Task] = None
        self._conn: Optional[libsql.Connection] = None
        
    def _ensure_connection(self):
        """Ensure libsql connection is established."""
        if not self._conn:
            if not self.sync_url:
                raise ValueError("No sync_url configured. Set TURSO_DATABASE_URL environment variable.")
            
            # Create connection with sync capabilities
            self._conn = libsql.connect(
                self.db_path,
                sync_url=self.sync_url,
                auth_token=self.auth_token,
                sync_interval=0  # We'll handle sync ourselves
            )
            logger.info(f"Connected to embedded database at {self.db_path}")
            
    async def sync_once(self) -> bool:
        """Perform a single sync operation.
        
        Returns:
            True if sync succeeded, False otherwise
        """
        try:
            self._ensure_connection()
            await asyncio.to_thread(self._conn.sync)
            logger.info("Successfully synced with remote Turso database")
            return True
        except Exception as e:
            logger.warning(f"Sync failed: {e}")
            return False
            
    async def keep_fresh(self):
        """Background task to continuously sync the replica."""
        logger.info(f"Starting replica sync with {self.sync_interval}s interval")
        
        while True:
            try:
                await self.sync_once()
            except asyncio.CancelledError:
                logger.info("Replica sync task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in sync loop: {e}")
                
            await asyncio.sleep(self.sync_interval)
            
    def start_background_sync(self) -> asyncio.Task:
        """Start the background sync task.
        
        Returns:
            The created asyncio Task
        """
        if self._sync_task and not self._sync_task.done():
            logger.warning("Sync task already running")
            return self._sync_task
            
        self._sync_task = asyncio.create_task(self.keep_fresh())
        return self._sync_task
        
    async def stop(self):
        """Stop the background sync task and close connection."""
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
                
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Closed replica sync connection")


# Global replica sync instance
_replica_sync: Optional[ReplicaSync] = None


def get_replica_sync() -> Optional[ReplicaSync]:
    """Get the global replica sync instance."""
    return _replica_sync


async def start_replica_sync(
    db_path: str = "embedded.db",
    sync_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    sync_interval: Optional[int] = None
) -> ReplicaSync:
    """Start global replica sync.
    
    Args:
        db_path: Path to local embedded database
        sync_url: Remote Turso URL (defaults to env var)
        auth_token: Auth token (defaults to env var)
        sync_interval: Sync interval in seconds (defaults to env var)
        
    Returns:
        The ReplicaSync instance
    """
    global _replica_sync
    
    if _replica_sync:
        logger.warning("Replica sync already started")
        return _replica_sync
        
    _replica_sync = ReplicaSync(
        db_path=db_path,
        sync_url=sync_url,
        auth_token=auth_token,
        sync_interval=sync_interval
    )
    
    # Perform initial sync
    await _replica_sync.sync_once()
    
    # Start background sync
    _replica_sync.start_background_sync()
    
    return _replica_sync


async def stop_replica_sync():
    """Stop the global replica sync."""
    global _replica_sync
    
    if _replica_sync:
        await _replica_sync.stop()
        _replica_sync = None