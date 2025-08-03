#!/usr/bin/env python3
"""
Interactive TUI Dashboard for Synth AI experiments.

Launch with: python -m synth_ai.tui.dashboard
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, DataTable, Static, Input, Button, 
    TabbedContent, TabPane, Label, ProgressBar
)
from textual.reactive import reactive
from textual.binding import Binding
from textual import on
from textual.timer import Timer

from ..tracing_v3.turso.manager import AsyncSQLTraceManager

class ExperimentRow:
    """Data structure for experiment display."""
    def __init__(self, exp_id: str, name: str, description: str, 
                 created_at: datetime, sessions: int, events: int, 
                 messages: int, cost: float, tokens: int):
        self.exp_id = exp_id
        self.name = name or "Unnamed"
        self.description = description or ""
        self.created_at = created_at
        self.sessions = sessions
        self.events = events  
        self.messages = messages
        self.cost = cost
        self.tokens = tokens
        
    def to_row(self) -> List[str]:
        """Convert to table row format."""
        return [
            self.exp_id[:8],  # Shortened ID
            self.name[:20],   # Truncated name
            str(self.sessions),
            str(self.events),
            str(self.messages),
            f"${self.cost:.4f}",
            f"{self.tokens:,}",
            self.created_at.strftime("%H:%M")
        ]

class ExperimentTable(DataTable):
    """Custom DataTable for experiments with refresh capability."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.experiments: List[ExperimentRow] = []
        self.selected_exp_id: Optional[str] = None
        
    def setup_table(self):
        """Initialize table columns."""
        self.add_columns(
            "ID", "Name", "Sessions", "Events", 
            "Messages", "Cost", "Tokens", "Time"
        )
        
    async def refresh_data(self, db_manager: AsyncSQLTraceManager):
        """Refresh experiment data from database."""
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
                        exp_id=row['experiment_id'],
                        name=row['name'],
                        description=row['description'],
                        created_at=row['created_at'],
                        sessions=int(row['num_sessions'] or 0),
                        events=int(row['num_events'] or 0),
                        messages=int(row['num_messages'] or 0),
                        cost=float(row['total_cost'] or 0.0),
                        tokens=int(row['total_tokens'] or 0)
                    )
                    self.experiments.append(exp_row)
                    self.add_row(*exp_row.to_row(), key=exp_row.exp_id)
                
        except Exception as e:
            logging.error(f"Failed to refresh experiments: {e}")
            
    def get_selected_experiment(self) -> Optional[ExperimentRow]:
        """Get currently selected experiment."""
        if self.cursor_row >= 0 and self.cursor_row < len(self.experiments):
            return self.experiments[self.cursor_row]
        return None

class ExperimentDetail(Static):
    """Detailed view of selected experiment."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_experiment: Optional[ExperimentRow] = None
        
    def update_experiment(self, experiment: Optional[ExperimentRow]):
        """Update the displayed experiment details."""
        self.current_experiment = experiment
        if experiment:
            details = f"""
🔬 **{experiment.name}**
ID: {experiment.exp_id}
Description: {experiment.description or 'No description'}

📊 **Statistics**
Sessions: {experiment.sessions}
Events: {experiment.events}  
Messages: {experiment.messages}
Cost: ${experiment.cost:.4f}
Tokens: {experiment.tokens:,}

🕒 **Created**: {experiment.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
        else:
            details = "Select an experiment to view details"
            
        self.update(details)

class DatabaseStatus(Static):
    """Display database connection status."""
    
    connection_status = reactive("🔴 Disconnected")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def render(self) -> str:
        return f"Database: {self.connection_status}"
        
    def set_connected(self, url: str):
        parsed = urlparse(url)
        host_info = f"{parsed.hostname}:{parsed.port}" if parsed.port else str(parsed.hostname)
        self.connection_status = f"🟢 Connected ({host_info})"
        
    def set_disconnected(self, error: str = ""):
        error_text = f" - {error}" if error else ""
        self.connection_status = f"🔴 Disconnected{error_text}"

class SynthDashboard(App):
    """Main Synth AI TUI Dashboard application."""
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-gutter: 1;
    }
    
    #header {
        column-span: 2;
        height: 3;
    }
    
    #experiments-table {
        row-span: 2;
    }
    
    #experiment-detail {
        height: 1fr;
    }
    
    #status-bar {
        column-span: 2;
        height: 3;
    }
    
    ExperimentTable {
        height: 100%;
    }
    
    ExperimentDetail {
        border: solid $primary;
        padding: 1;
        height: 100%;
    }
    
    DatabaseStatus {
        height: 1;
        padding: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("d", "toggle_debug", "Debug"),
        ("ctrl+c", "quit", "Quit"),
    ]
    
    def __init__(self, db_url: str = "sqlite+aiosqlite:///./synth_ai.db/dbs/default/data"):
        super().__init__()
        self.db_url = db_url
        self.db_manager: Optional[AsyncSQLTraceManager] = None
        self.refresh_timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)
        
        with Container(id="experiments-table"):
            yield Static("🧪 Experiments", classes="section-title")
            yield ExperimentTable(id="experiments")
            
        with Container(id="experiment-detail"):
            yield Static("📋 Details", classes="section-title")  
            yield ExperimentDetail(id="detail")
            
        with Container(id="status-bar"):
            yield DatabaseStatus(id="db-status")
            yield Footer()
            
    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Setup database connection
        try:
            self.db_manager = AsyncSQLTraceManager(self.db_url)
            await self.db_manager.initialize()
            
            db_status = self.query_one("#db-status", DatabaseStatus)
            db_status.set_connected(self.db_url)
            
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            db_status = self.query_one("#db-status", DatabaseStatus)
            db_status.set_disconnected(str(e))
            
        # Setup experiment table
        exp_table = self.query_one("#experiments", ExperimentTable)
        exp_table.setup_table()
        
        # Initial data load
        await self.action_refresh()
        
        # Start auto-refresh timer (every 5 seconds)
        self.refresh_timer = self.set_interval(5.0, self._auto_refresh)
        
    async def _auto_refresh(self) -> None:
        """Auto-refresh data periodically."""
        if self.db_manager:
            exp_table = self.query_one("#experiments", ExperimentTable)
            await exp_table.refresh_data(self.db_manager)
            
    async def action_refresh(self) -> None:
        """Manual refresh action."""
        if self.db_manager:
            exp_table = self.query_one("#experiments", ExperimentTable)
            await exp_table.refresh_data(self.db_manager)
            
    async def action_quit(self) -> None:
        """Quit the application."""
        if self.refresh_timer:
            self.refresh_timer.stop()
        if self.db_manager:
            await self.db_manager.close()
        self.exit()
        
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

def main():
    """Main entry point for the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Synth AI Interactive Dashboard")
    parser.add_argument(
        "-u", "--url", 
        default="sqlite+libsql://http://127.0.0.1:8080",
        help="Database URL (default: sqlite+libsql://http://127.0.0.1:8080)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        
    # Run the dashboard
    app = SynthDashboard(db_url=args.url)
    app.run()

if __name__ == "__main__":
    main()