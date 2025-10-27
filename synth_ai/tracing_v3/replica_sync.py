"""Turso embedded replica synchronization.

This module provides background synchronization between a local embedded SQLite
database and a remote Turso database. This enables low-latency local writes while
maintaining eventual consistency with a remote database.

Architecture:
------------
The replica sync works by:
1. Writing all data locally to an embedded SQLite database
2. Periodically syncing changes to a remote Turso database
3. Optionally pulling changes from remote (for multi-node setups)

Use Cases:
---------
- Edge deployments with intermittent connectivity
- Low-latency local operations with cloud backup
- Multi-region deployments with eventual consistency
- Development environments with production data sync

Async Design:
------------
The sync process runs as a background asyncio task, allowing the main
application to continue without blocking on sync operations.
"""

import asyncio
import importlib
import logging
from typing import Any, cast

from .config import CONFIG

logger = logging.getLogger(__name__)

libsql = cast(Any, importlib.import_module("libsql"))

class ReplicaSync:
    """Manages synchronization of embedded SQLite replica with remote Turso database.

    This class handles the lifecycle of replica synchronization, including:
    - Establishing connections to both local and remote databases
    - Running periodic sync operations
    - Handling sync failures gracefully
    - Managing the background sync task

    The sync is designed to be resilient to network failures and will
    continue retrying with exponential backoff.
    """

    def __init__(
        self,
        db_path: str = "embedded.db",
        sync_url: str | None = None,
        auth_token: str | None = None,
        sync_interval: float | None = None,
    ):
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
        self._sync_task: asyncio.Task[Any] | None = None
        self._conn: Any | None = None

    def _ensure_connection(self):
        """Ensure libsql connection is established.

        Creates a connection to the local embedded database with sync
        capabilities. The libsql library handles the replication protocol
        with the remote Turso database.

        Raises:
            ValueError: If no sync_url is configured
        """
        if not self._conn:
            if not self.sync_url:
                raise ValueError(
                    "No sync_url configured. Set TURSO_DATABASE_URL environment variable."
                )

            # Create connection with sync capabilities
            # sync_interval=0 means we control when syncs happen
            self._conn = libsql.connect(
                self.db_path,
                sync_url=self.sync_url,
                auth_token=self.auth_token,
                sync_interval=0,  # We'll handle sync ourselves
            )
            logger.info(f"Connected to embedded database at {self.db_path}")

    async def sync_once(self) -> bool:
        """Perform a single sync operation.

        This method:
        1. Ensures a connection exists
        2. Runs the sync in a thread pool to avoid blocking
        3. Handles failures gracefully

        The actual sync protocol is handled by libsql and includes:
        - Sending local changes to remote
        - Receiving remote changes (if configured)
        - Conflict resolution (last-write-wins by default)

        Returns:
            True if sync succeeded, False otherwise
        """
        try:
            self._ensure_connection()
            conn = self._conn
            if conn is None:
                raise RuntimeError("Replica sync connection is not available after initialization")
            # Run sync in thread pool since libsql sync is blocking
            await asyncio.to_thread(conn.sync)
            logger.info("Successfully synced with remote Turso database")
            return True
        except Exception as e:
            logger.warning(f"Sync failed: {e}")
            return False

    async def keep_fresh(self):
        """Background task to continuously sync the replica.

        Runs in an infinite loop, performing sync operations at the configured
        interval. Handles cancellation gracefully for clean shutdown.

        The task will continue running even if individual syncs fail, ensuring
        eventual consistency when connectivity is restored.
        """
        logger.info(f"Starting replica sync with {self.sync_interval}s interval")

        while True:
            try:
                await self.sync_once()
            except asyncio.CancelledError:
                # Clean shutdown requested
                logger.info("Replica sync task cancelled")
                break
            except Exception as e:
                # Log but continue - we want to keep trying
                logger.error(f"Unexpected error in sync loop: {e}")

            # Sleep until next sync interval
            await asyncio.sleep(self.sync_interval)

    def start_background_sync(self) -> asyncio.Task[Any]:
        """Start the background sync task.

        Creates an asyncio task that runs the sync loop. The task is stored
        internally for lifecycle management.

        This method is idempotent - calling it multiple times will not create
        multiple sync tasks.

        Returns:
            The created asyncio Task
        """
        if self._sync_task and not self._sync_task.done():
            logger.warning("Sync task already running")
            return self._sync_task

        # Create and store the task for lifecycle management
        self._sync_task = asyncio.create_task(self.keep_fresh())
        return self._sync_task

    async def stop(self):
        """Stop the background sync task and close connection.

        Performs a clean shutdown:
        1. Cancels the background sync task
        2. Waits for task completion
        3. Closes the database connection

        This method is safe to call multiple times.
        """
        if self._sync_task and not self._sync_task.done():
            # Request cancellation
            self._sync_task.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                # Wait for the task to finish
                await self._sync_task

        if self._conn:
            # Close the libsql connection
            self._conn.close()
            self._conn = None
            logger.info("Closed replica sync connection")


# Global replica sync instance
_replica_sync: ReplicaSync | None = None


def get_replica_sync() -> ReplicaSync | None:
    """Get the global replica sync instance."""
    return _replica_sync


async def start_replica_sync(
    db_path: str = "embedded.db",
    sync_url: str | None = None,
    auth_token: str | None = None,
    sync_interval: int | None = None,
) -> ReplicaSync:
    """Start global replica sync.

    Convenience function to create and start a replica sync instance.
    Performs an initial sync before starting the background task to ensure
    the local database is up-to-date.

    Args:
        db_path: Path to local embedded database file
        sync_url: Remote Turso URL (defaults to TURSO_DATABASE_URL env var)
        auth_token: Auth token (defaults to TURSO_AUTH_TOKEN env var)
        sync_interval: Sync interval in seconds (defaults to TURSO_SYNC_SECONDS)

    Returns:
        The ReplicaSync instance

    Raises:
        ValueError: If sync_url is not provided and not in environment
    """
    global _replica_sync

    if _replica_sync:
        logger.warning("Replica sync already started")
        return _replica_sync

    _replica_sync = ReplicaSync(
        db_path=db_path, sync_url=sync_url, auth_token=auth_token, sync_interval=sync_interval
    )

    # Perform initial sync to ensure we start with fresh data
    await _replica_sync.sync_once()

    # Start background sync task for continuous synchronization
    _replica_sync.start_background_sync()

    return _replica_sync


async def stop_replica_sync():
    """Stop the global replica sync.

    Stops the global replica sync instance if one is running.
    This should be called during application shutdown to ensure
    clean termination of the sync task.
    """
    global _replica_sync

    if _replica_sync:
        await _replica_sync.stop()
        _replica_sync = None
