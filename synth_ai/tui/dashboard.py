#!/usr/bin/env python3
"""
Interactive TUI Dashboard for Synth AI experiments.

Launch with: python -m synth_ai.tui.dashboard
"""

import logging
import os
from datetime import datetime
from urllib.parse import urlparse

# Import textual components with graceful fallback
try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container
    from textual.reactive import reactive
    from textual.timer import Timer
    from textual.widgets import (
        DataTable,
        Footer,
        Header,
        Static,
    )
    _TEXTUAL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Textual not available - provide dummy classes for type checking
    on = None  # type: ignore
    App = object  # type: ignore
    ComposeResult = object  # type: ignore
    Binding = object  # type: ignore
    Container = object  # type: ignore
    reactive = lambda x: x  # type: ignore
    Timer = object  # type: ignore
    DataTable = object  # type: ignore
    Footer = object  # type: ignore
    Header = object  # type: ignore
    Static = object  # type: ignore
    _TEXTUAL_AVAILABLE = False

# Import database manager with graceful fallback
try:
    from synth_ai.tracing_v3.turso.native_manager import NativeLibsqlTraceManager  # type: ignore[import-untyped]
    _DB_AVAILABLE = True
except (ImportError, ModuleNotFoundError, TypeError):
    # Database manager not available - provide dummy class
    NativeLibsqlTraceManager = object  # type: ignore
    _DB_AVAILABLE = False

import asyncio
import requests
from datetime import timedelta


class ExperimentRow:
    """Data structure for experiment display."""

    def __init__(
        self,
        exp_id: str,
        name: str,
        description: str,
        created_at: datetime,
        sessions: int,
        events: int,
        messages: int,
        cost: float,
        tokens: int,
    ):
        self.exp_id = exp_id
        self.name = name or "Unnamed"
        self.description = description or ""
        self.created_at = created_at
        self.sessions = sessions
        self.events = events
        self.messages = messages
        self.cost = cost
        self.tokens = tokens

    def to_row(self) -> list[str]:
        """Convert to table row format."""
        return [
            self.exp_id[:8],  # Shortened ID
            self.name[:20],  # Truncated name
            str(self.sessions),
            str(self.events),
            str(self.messages),
            f"${self.cost:.4f}",
            f"{self.tokens:,}",
            self.created_at.strftime("%H:%M"),
        ]


class ExperimentTable(DataTable):
    """Custom DataTable for experiments with refresh capability."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.experiments: list[ExperimentRow] = []
        self.selected_exp_id: str | None = None

    def setup_table(self):
        """Initialize table columns."""
        self.add_columns("ID", "Name", "Sessions", "Events", "Messages", "Cost", "Tokens", "Time")

    async def refresh_data(self, db_manager: NativeLibsqlTraceManager | None) -> None:
        """Refresh experiment data from database."""
        if not db_manager:
            # Database not available, clear the table
            self.experiments.clear()
            self.clear()
            return

        try:
            # Get experiment list with stats using raw query
            df = await db_manager.query_traces("""
                SELECT 
                    e.experiment_id,
                    e.name,
                    e.description,
                    e.created_at,
                    COUNT(DISTINCT st.session_id) as num_sessions,
                    COUNT(DISTINCT ev.id) as num_events,
                    COUNT(DISTINCT m.id) as num_messages,
                    SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost,
                    SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) as total_tokens
                FROM experiments e
                LEFT JOIN session_traces st ON e.experiment_id = st.experiment_id
                LEFT JOIN events ev ON st.session_id = ev.session_id
                LEFT JOIN messages m ON st.session_id = m.session_id
                GROUP BY e.experiment_id, e.name, e.description, e.created_at
                ORDER BY e.created_at DESC
            """)

            self.experiments.clear()
            self.clear()

            if not df.empty:
                for _, row in df.iterrows():
                    exp_row = ExperimentRow(
                        exp_id=row["experiment_id"],
                        name=row["name"],
                        description=row["description"],
                        created_at=row["created_at"],
                        sessions=int(row["num_sessions"] or 0),
                        events=int(row["num_events"] or 0),
                        messages=int(row["num_messages"] or 0),
                        cost=float(row["total_cost"] or 0.0),
                        tokens=int(row["total_tokens"] or 0),
                    )
                    self.experiments.append(exp_row)
                    self.add_row(*exp_row.to_row(), key=exp_row.exp_id)

        except Exception as e:
            logging.error(f"Failed to refresh experiments: {e}")

    def get_selected_experiment(self) -> ExperimentRow | None:
        """Get currently selected experiment."""
        if self.cursor_row >= 0 and self.cursor_row < len(self.experiments):
            return self.experiments[self.cursor_row]
        return None


class ExperimentDetail(Static):
    """Detailed view of selected experiment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_experiment: ExperimentRow | None = None

    def update_experiment(self, experiment: ExperimentRow | None):
        """Update the displayed experiment details."""
        self.current_experiment = experiment
        if experiment:
            details = f"""
🔬 **{experiment.name}**
ID: {experiment.exp_id}
Description: {experiment.description or "No description"}

📊 **Statistics**
Sessions: {experiment.sessions}
Events: {experiment.events}  
Messages: {experiment.messages}
Cost: ${experiment.cost:.4f}
Tokens: {experiment.tokens:,}

🕒 **Created**: {experiment.created_at.strftime("%Y-%m-%d %H:%M:%S")}
            """.strip()
        else:
            details = "Select an experiment to view details"

        self.update(details)


class DatabaseStatus(Static):
    """Display database connection status."""

    connection_status = reactive("🔴 Disconnected")
    db_info = reactive("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def render(self) -> str:
        status_line = f"Database: {self.connection_status}"
        if self.db_info:
            status_line += f" | {self.db_info}"
        return status_line

    def set_connected(self, url: str, db_name: str = ""):
        parsed = urlparse(url)
        if "sqlite" in url:
            # Extract just the filename for cleaner display
            from pathlib import Path
            try:
                path_part = url.split("///")[-1]
                filename = Path(path_part).name
                self.connection_status = f"🟢 {filename}"
            except:
                self.connection_status = f"🟢 Connected"
        else:
            host_info = f"{parsed.hostname}:{parsed.port}" if parsed.port else str(parsed.hostname)
            self.connection_status = f"🟢 {host_info}"
        
        if db_name:
            self.db_info = f"[{db_name}]"

    def set_disconnected(self, error: str = ""):
        error_text = f" - {error}" if error else ""
        self.connection_status = f"🔴 Disconnected{error_text}"
        self.db_info = ""
    
    def set_db_selector(self, current: int, total: int):
        """Show database selector info."""
        if total > 1:
            self.db_info = f"DB {current + 1}/{total} (n/p to switch)"
        else:
            self.db_info = ""


class BalanceStatus(Static):
    """Display balance and spending information (local + global)."""

    # Global (backend API)
    global_balance = reactive("$0.00")
    global_spend_24h = reactive("$0.00")
    global_spend_7d = reactive("$0.00")
    global_status = reactive("⏳")
    
    # Local (database)
    local_traces = reactive(0)
    local_cost = reactive("$0.00")
    local_tokens = reactive(0)
    local_tasks = reactive([])  # List of (task_name, count) tuples
    local_status = reactive("⏳")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def render(self) -> str:
        # Format tokens safely
        if isinstance(self.local_tokens, int) and self.local_tokens > 0:
            if self.local_tokens >= 1_000_000:
                tokens_str = f"{self.local_tokens / 1_000_000:.1f}M"
            elif self.local_tokens >= 1_000:
                tokens_str = f"{self.local_tokens / 1_000:.1f}K"
            else:
                tokens_str = f"{self.local_tokens}"
        else:
            tokens_str = str(self.local_tokens)
        
        # Format tasks - show top 3 only
        tasks_str = ""
        if self.local_tasks and len(self.local_tasks) > 0:
            top_tasks = self.local_tasks[:3]
            task_lines = [f"{name} ({count})" for name, count in top_tasks]
            tasks_str = " | " + ", ".join(task_lines)
            if len(self.local_tasks) > 3:
                tasks_str += f", +{len(self.local_tasks) - 3}"
        
        # Compact single-line format
        return f"""[b]Local[/b] {self.local_status} {self.local_traces} traces | {self.local_cost} | {tokens_str} tokens{tasks_str}

[b]Global[/b] {self.global_status} {self.global_balance} | 24h: {self.global_spend_24h} | 7d: {self.global_spend_7d}"""

    def update_global(self, balance: float, spend_24h: float, spend_7d: float):
        """Update global backend balance information."""
        self.global_balance = f"${balance:.2f}"
        self.global_spend_24h = f"${spend_24h:.2f}"
        self.global_spend_7d = f"${spend_7d:.2f}"
        self.global_status = "✅"
    
    def update_local(self, traces: int, cost: float, tokens: int, tasks: list[tuple[str, int]] | None = None):
        """Update local database statistics."""
        self.local_traces = traces
        self.local_cost = f"${cost:.4f}"
        self.local_tokens = tokens
        self.local_tasks = tasks or []
        self.local_status = "✅"

    def set_global_loading(self):
        """Show loading state for global data."""
        self.global_balance = "..."
        self.global_spend_24h = "..."
        self.global_spend_7d = "..."
        self.global_status = "⏳"
    
    def set_local_loading(self):
        """Show loading state for local data."""
        self.local_traces = 0
        self.local_cost = "..."
        self.local_tokens = 0
        self.local_tasks = []
        self.local_status = "⏳"

    def set_global_error(self, error: str):
        """Show error state for global data."""
        self.global_balance = f"Error"
        self.global_spend_24h = "-"
        self.global_spend_7d = "-"
        self.global_status = f"❌"
        
    def set_local_error(self, error: str):
        """Show error state for local data."""
        self.local_traces = 0
        self.local_cost = "Error"
        self.local_tokens = 0
        self.local_tasks = []
        self.local_status = f"❌"
    
    def set_global_unavailable(self):
        """Mark global data as unavailable (no API key)."""
        self.global_balance = "N/A"
        self.global_spend_24h = "N/A"
        self.global_spend_7d = "N/A"
        self.global_status = "⚪"


class ActiveRunsTable(DataTable):
    """Display currently active/running sessions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_runs: list[dict] = []

    def setup_table(self):
        """Initialize table columns."""
        self.add_columns("Session", "Experiment", "Started", "Duration", "Events", "Status")

    async def refresh_data(self, db_manager: NativeLibsqlTraceManager | None) -> None:
        """Refresh active runs data from database."""
        if not db_manager:
            # Database not available, clear the table
            self.active_runs.clear()
            self.clear()
            return

        try:
            # Get active sessions (those with recent activity in last 5 minutes)
            cutoff_time = datetime.now() - timedelta(minutes=5)

            df = await db_manager.query_traces("""
                WITH recent_sessions AS (
                    SELECT
                        st.session_id,
                        st.experiment_id,
                        st.created_at,
                        e.name as experiment_name,
                        COUNT(ev.id) as event_count,
                        MAX(ev.created_at) as last_event_time
                    FROM session_traces st
                    LEFT JOIN experiments e ON st.experiment_id = e.experiment_id
                    LEFT JOIN events ev ON st.session_id = ev.session_id
                    WHERE st.created_at >= :cutoff_time
                    GROUP BY st.session_id, st.experiment_id, st.created_at, e.name
                )
                SELECT
                    session_id,
                    experiment_id,
                    experiment_name,
                    created_at,
                    event_count,
                    last_event_time
                FROM recent_sessions
                ORDER BY last_event_time DESC
            """, {"cutoff_time": cutoff_time})

            self.active_runs.clear()
            self.clear()

            if not df.empty:
                for _, row in df.iterrows():
                    session_id = str(row["session_id"])
                    experiment_name = row["experiment_name"] or "Unknown"
                    
                    # Parse datetime strings
                    try:
                        if isinstance(row["created_at"], str):
                            from dateutil import parser as date_parser
                            started_at = date_parser.parse(row["created_at"])
                        else:
                            started_at = row["created_at"]
                        
                        if isinstance(row["last_event_time"], str):
                            from dateutil import parser as date_parser
                            last_event_time = date_parser.parse(row["last_event_time"])
                        else:
                            last_event_time = row["last_event_time"]
                    except Exception as e:
                        logging.error(f"Failed to parse datetime: {e}")
                        continue
                    
                    duration = datetime.now() - started_at

                    # Format duration
                    if duration.total_seconds() < 3600:  # Less than 1 hour
                        duration_str = f"{int(duration.total_seconds() // 60)}m"
                    else:
                        hours = int(duration.total_seconds() // 3600)
                        minutes = int((duration.total_seconds() % 3600) // 60)
                        duration_str = f"{hours}h {minutes}m"

                    # Status based on recent activity
                    time_since_last = datetime.now() - last_event_time
                    if time_since_last.total_seconds() < 60:  # Active in last minute
                        status = "🟢 Active"
                    elif time_since_last.total_seconds() < 300:  # Active in last 5 minutes
                        status = "🟡 Recent"
                    else:
                        status = "🟠 Idle"

                    run_info = {
                        "session_id": session_id,
                        "experiment_name": experiment_name,
                        "started_at": started_at,
                        "duration": duration_str,
                        "events": int(row["event_count"]),
                        "status": status
                    }
                    self.active_runs.append(run_info)
                    self.add_row(
                        session_id[:8],  # Shortened session ID
                        experiment_name[:20],  # Truncated name
                        started_at.strftime("%H:%M:%S"),
                        duration_str,
                        str(run_info["events"]),
                        status,
                        key=session_id
                    )

        except Exception as e:
            logging.error(f"Failed to refresh active runs: {e}")


def find_databases() -> list[tuple[str, str]]:
    """Find all available databases in common locations.
    
    Returns:
        List of (name, path) tuples
    """
    databases = []
    search_paths = [
        "traces/v3",
        "traces",
        ".",
    ]
    
    for search_path in search_paths:
        try:
            from pathlib import Path
            search_dir = Path(search_path)
            if not search_dir.exists():
                continue
                
            # Find all .db files
            for db_file in search_dir.glob("**/*.db"):
                if db_file.is_file():
                    # Use relative path from current directory
                    rel_path = str(db_file.relative_to(Path.cwd()))
                    # Create a friendly name
                    name = db_file.stem  # filename without .db
                    if len(databases) == 0:
                        name = f"{name} (default)"
                    databases.append((name, rel_path))
        except Exception as e:
            logging.debug(f"Error scanning {search_path}: {e}")
    
    # If no databases found, return default
    if not databases:
        databases.append(("synth_ai (default)", "traces/v3/synth_ai.db"))
    
    return databases


class SynthDashboard(App if _TEXTUAL_AVAILABLE else object):
    """Main Synth AI TUI Dashboard application."""

    CSS = """
    Screen {
        layout: grid;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: auto 1fr 1fr auto;
        grid-gutter: 1;
    }

    #header {
        column-span: 3;
        height: 3;
    }

    #experiments-table {
        row-span: 2;
    }

    #active-runs-panel {
        column-span: 1;
    }

    #balance-status {
        column-span: 1;
    }

    #experiment-detail {
        column-span: 2;
        height: 1fr;
    }

    #status-bar {
        column-span: 3;
        height: 3;
    }

    ExperimentTable {
        height: 100%;
    }

    ActiveRunsTable {
        height: 100%;
    }

    ExperimentDetail {
        border: solid $primary;
        padding: 1;
        height: 100%;
    }

    BalanceStatus {
        border: solid $primary;
        padding: 1;
        height: 100%;
    }

    DatabaseStatus {
        height: 1;
        padding: 0 1;
    }

    .section-title {
        text-style: bold;
        height: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("n", "next_db", "Next DB"),
        Binding("p", "prev_db", "Prev DB"),
        Binding("d", "toggle_debug", "Debug"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, db_url: str = "sqlite+aiosqlite:///traces/v3/synth_ai.db"):
        super().__init__()
        self.db_url = db_url
        self.db_manager: NativeLibsqlTraceManager | None = None
        self.refresh_timer: Timer | None = None
        
        # Database discovery and selection
        self.available_dbs: list[tuple[str, str]] = find_databases()
        self.current_db_index: int = 0
        
        # Log discovered databases
        logging.info(f"Found {len(self.available_dbs)} database(s):")
        for idx, (name, path) in enumerate(self.available_dbs):
            logging.info(f"  [{idx+1}] {name}: {path}")
        
        # Try to find the initial db_url in available_dbs
        for idx, (name, path) in enumerate(self.available_dbs):
            if path in db_url or db_url.endswith(path):
                self.current_db_index = idx
                logging.info(f"Using database: {name} ({path})")
                break

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)

        with Container(id="experiments-table"):
            yield Static("🧪 Experiments", classes="section-title")
            yield ExperimentTable(id="experiments")

        with Container(id="active-runs-panel"):
            yield Static("⚡ Active Runs", classes="section-title")
            yield ActiveRunsTable(id="active-runs")

        with Container(id="balance-status"):
            yield Static("💰 Balance & Stats", classes="section-title")
            yield BalanceStatus(id="balance")

        with Container(id="experiment-detail"):
            yield Static("📋 Details", classes="section-title")
            yield ExperimentDetail(id="detail")

        with Container(id="status-bar"):
            yield DatabaseStatus(id="db-status")
            yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Setup database connection - make it optional
        await self._connect_to_database()

        # Setup tables
        exp_table = self.query_one("#experiments", ExperimentTable)
        exp_table.setup_table()

        active_runs_table = self.query_one("#active-runs", ActiveRunsTable)
        active_runs_table.setup_table()

        # Set balance loading state
        balance_widget = self.query_one("#balance", BalanceStatus)
        balance_widget.set_global_loading()
        balance_widget.set_local_loading()

        # Initial data load
        await self.action_refresh()

        # Start auto-refresh timer (every 5 seconds)
        self.refresh_timer = self.set_interval(5.0, self._auto_refresh_data)

    async def _auto_refresh_data(self) -> None:
        """Auto-refresh data periodically."""
        exp_table = self.query_one("#experiments", ExperimentTable)
        active_runs_table = self.query_one("#active-runs", ActiveRunsTable)
        balance_widget = self.query_one("#balance", BalanceStatus)

        if self.db_manager:
            await exp_table.refresh_data(self.db_manager)
            await active_runs_table.refresh_data(self.db_manager)
            await self._refresh_local_stats(balance_widget)
        
        # Always try to refresh global balance (independent of local DB)
        await self._refresh_global_balance(balance_widget)

    async def action_refresh(self) -> None:
        """Manual refresh action."""
        exp_table = self.query_one("#experiments", ExperimentTable)
        active_runs_table = self.query_one("#active-runs", ActiveRunsTable)
        balance_widget = self.query_one("#balance", BalanceStatus)

        balance_widget.set_global_loading()
        balance_widget.set_local_loading()

        if self.db_manager:
            await exp_table.refresh_data(self.db_manager)
            await active_runs_table.refresh_data(self.db_manager)
            await self._refresh_local_stats(balance_widget)
        
        # Always try to refresh global balance (independent of local DB)
        await self._refresh_global_balance(balance_widget)

    async def _refresh_local_stats(self, balance_widget: BalanceStatus) -> None:
        """Refresh local database statistics."""
        if not self.db_manager:
            logging.warning("No database manager available for local stats")
            balance_widget.set_local_error("No database")
            return
        
        try:
            logging.info("Fetching local stats from database...")
            # Query local trace statistics
            df = await self.db_manager.query_traces("""
                SELECT 
                    COUNT(DISTINCT st.session_id) as num_traces,
                    SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost,
                    SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) as total_tokens
                FROM session_traces st
                LEFT JOIN events ev ON st.session_id = ev.session_id
            """)
            
            # Query task/environment breakdown from metadata
            task_df = await self.db_manager.query_traces("""
                SELECT 
                    json_extract(metadata, '$.env_name') as task_name,
                    COUNT(DISTINCT session_id) as trace_count
                FROM session_traces
                WHERE json_extract(metadata, '$.env_name') IS NOT NULL
                GROUP BY task_name
                ORDER BY trace_count DESC
                LIMIT 10
            """)
            
            if not df.empty:
                row = df.iloc[0]
                num_traces = int(row["num_traces"] or 0)
                total_cost = float(row["total_cost"] or 0.0)
                total_tokens = int(row["total_tokens"] or 0)
                
                # Parse task data
                tasks = []
                if not task_df.empty:
                    for _, task_row in task_df.iterrows():
                        task_name = task_row["task_name"]
                        count = int(task_row["trace_count"])
                        if task_name:
                            tasks.append((str(task_name), count))
                
                logging.info(f"Local stats: {num_traces} traces, ${total_cost:.4f}, {total_tokens} tokens, {len(tasks)} tasks")
                balance_widget.update_local(num_traces, total_cost, total_tokens, tasks)
            else:
                logging.warning("Query returned empty dataframe")
                balance_widget.update_local(0, 0.0, 0, [])
                
        except Exception as e:
            logging.error(f"Failed to refresh local stats: {e}", exc_info=True)
            balance_widget.set_local_error(str(e)[:20])

    async def _refresh_global_balance(self, balance_widget: BalanceStatus) -> None:
        """Refresh balance information from backend API."""
        try:
            # Try to get balance from environment or API
            api_key = os.getenv("SYNTH_API_KEY") or os.getenv("SYNTH_BACKEND_API_KEY")
            if not api_key:
                balance_widget.set_global_unavailable()
                return

            # Try to get backend URL from environment
            backend_url = os.getenv("SYNTH_BACKEND_BASE_URL") or "https://agent-learning.onrender.com/api/v1"

            # Fetch balance
            response = requests.get(
                f"{backend_url}/balance/current",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            balance = float(data.get("balance_dollars", 0.0))

            # Try to get usage data
            try:
                usage_response = requests.get(
                    f"{backend_url}/balance/usage/windows",
                    params={"hours": "24,168"},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=5
                )
                if usage_response.ok:
                    usage_data = usage_response.json()
                    windows = {
                        int(r.get("window_hours")): r
                        for r in usage_data.get("windows", [])
                        if isinstance(r.get("window_hours"), int)
                    }

                    spend_24h = 0.0
                    spend_7d = 0.0

                    if 24 in windows:
                        spend_24h = float(windows[24].get("total_spend_cents", 0)) / 100.0
                    if 168 in windows:
                        spend_7d = float(windows[168].get("total_spend_cents", 0)) / 100.0

                    balance_widget.update_global(balance, spend_24h, spend_7d)
                else:
                    # Fallback to just balance
                    balance_widget.update_global(balance, 0.0, 0.0)
            except Exception:
                # Fallback to just balance
                balance_widget.update_global(balance, 0.0, 0.0)

        except Exception as e:
            # Only show error if it's not just "endpoint not available"
            error_msg = str(e)
            if "500" in error_msg or "Internal Server Error" in error_msg:
                # Backend endpoint not implemented yet
                balance_widget.set_global_unavailable()
            else:
                balance_widget.set_global_error(error_msg[:30])

    async def action_quit(self) -> None:
        """Quit the application."""
        if self.refresh_timer:
            self.refresh_timer.stop()
        if self.db_manager:
            await self.db_manager.close()
        self.exit()

    async def _connect_to_database(self) -> None:
        """Connect to the current database."""
        db_status = self.query_one("#db-status", DatabaseStatus)
        balance_widget = self.query_one("#balance", BalanceStatus)
        
        try:
            # Close existing connection if any
            if self.db_manager:
                await self.db_manager.close()
                self.db_manager = None
            
            # Get current database info
            db_name, db_path = self.available_dbs[self.current_db_index]
            self.db_url = f"sqlite+aiosqlite:///{db_path}"
            
            logging.info(f"Connecting to database: {db_name} ({db_path})")
            
            self.db_manager = NativeLibsqlTraceManager(self.db_url)
            if self.db_manager:
                await self.db_manager.initialize()
                db_status.set_connected(self.db_url, db_name)
                db_status.set_db_selector(self.current_db_index, len(self.available_dbs))
                
                # Immediately refresh local stats after connecting
                logging.info("Refreshing local stats after connection...")
                await self._refresh_local_stats(balance_widget)
            else:
                db_status.set_disconnected("Database manager not available")
                balance_widget.set_local_error("No manager")
        except (ImportError, ModuleNotFoundError):
            # Database dependencies not available
            db_status.set_disconnected("Database dependencies missing (libsql)")
            self.db_manager = None
            balance_widget.set_local_error("No libsql")
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}", exc_info=True)
            db_status.set_disconnected(str(e))
            self.db_manager = None
            balance_widget.set_local_error(str(e)[:15])

    async def action_next_db(self) -> None:
        """Switch to next database."""
        if len(self.available_dbs) <= 1:
            return
        
        self.current_db_index = (self.current_db_index + 1) % len(self.available_dbs)
        await self._connect_to_database()
        await self.action_refresh()

    async def action_prev_db(self) -> None:
        """Switch to previous database."""
        if len(self.available_dbs) <= 1:
            return
        
        self.current_db_index = (self.current_db_index - 1) % len(self.available_dbs)
        await self._connect_to_database()
        await self.action_refresh()

    def action_toggle_debug(self) -> None:
        """Toggle debug mode."""
        # Could add debug panel or logging level toggle
        pass

    @on(DataTable.RowHighlighted, "#experiments")
    def on_experiment_selected(self, event: DataTable.RowHighlighted) -> None:
        """Handle experiment selection."""
        exp_table = self.query_one("#experiments", ExperimentTable)
        selected_exp = exp_table.get_selected_experiment()

        detail_panel = self.query_one("#detail", ExperimentDetail)
        detail_panel.update_experiment(selected_exp)


def main(argv: list[str] | None = None):
    """Main entry point for the dashboard."""
    # Check if textual is available
    if not _TEXTUAL_AVAILABLE:
        print("❌ Textual library is not available. Please install it with: pip install textual")
        return

    import argparse
    import os

    parser = argparse.ArgumentParser(description="Synth AI Interactive Dashboard")
    parser.add_argument(
        "-u",
        "--url",
        default=os.getenv("TUI_DB_URL", "sqlite+aiosqlite:///traces/v3/synth_ai.db"),
        help="Database URL (default: traces/v3/synth_ai.db)",
    )
    parser.add_argument("--debug", action="store_true", default=bool(os.getenv("TUI_DEBUG")), help="Enable debug logging")

    args = parser.parse_args(argv)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # Run the dashboard
    app = SynthDashboard(db_url=args.url)
    app.run()


if __name__ == "__main__":
    main()
