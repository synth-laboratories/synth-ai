#!/usr/bin/env python3
"""Live terminal dashboard for GEPA optimization runs using Rich + Plotille.

Usage:
    python gepa_tui.py --log /path/to/log.txt
    tail -f run.log | python gepa_tui.py
"""

import argparse
import io
import json
import os
import re
import sys
import time
import math
import datetime as dt
from typing import List, Optional, Deque, Tuple
from collections import deque

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich import box
    import plotille
except ImportError:
    print("ERROR: Required packages not installed. Install with:")
    print("  pip install rich plotille")
    sys.exit(1)

# ------------ Parsing helpers ------------

BEST_SCORE_EQ_RE = re.compile(r"best_score\s*=\s*([0-9]*\.?[0-9]+)")
BEST_SCORE_JSON_RE = re.compile(r'"best_score"\s*:\s*([0-9]*\.?[0-9]+)')
VAL_ACC_RE = re.compile(r"Validation Accuracy.*?:\s*([0-9]*\.?[0-9]+)")
USD_RE = re.compile(r'"total_usd"\s*:\s*([0-9]*\.?[0-9]+)')
BILLED_RE = re.compile(r'billed \$([0-9]*\.?[0-9]+)')
TRIAL_RESULTS_RE = re.compile(r'"mean"\s*:\s*([0-9]*\.?[0-9]+)')

def try_json_num(line: str, key: str) -> Optional[float]:
    try:
        obj = json.loads(line.strip())
        v = obj.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    except Exception:
        pass
    return None

# ------------ Model for dashboard state ------------

class State:
    def __init__(self, max_events: int = 10):
        self.start = time.time()
        self.best_scores: List[Tuple[float, float]] = []  # (t, best_score)
        self.val_acc: Optional[float] = None
        self.cost_usd: float = 0.0
        self.events: Deque[str] = deque(maxlen=max_events)
        self.filename: Optional[str] = None
        self.status: str = "running"
        self.trial_count: int = 0

    def add_event(self, msg: str):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{ts}] {msg}")

    def push_best(self, v: float):
        t = time.time() - self.start
        # dedupe if unchanged from last
        if not self.best_scores or abs(self.best_scores[-1][1] - v) > 1e-12:
            self.best_scores.append((t, v))

# ------------ Rendering ------------

def sparkline(y: List[float]) -> str:
    """Quick sparkline using plotille."""
    if len(y) < 2:
        return "•"
    f = plotille.Figure()
    f.width = 50
    f.height = 12
    f.color_mode = None  # let Rich color the panel; keep ASCII clean
    f.set_x_limits(min_=0, max_=len(y) - 1)
    f.set_y_limits(min_=min(y) - 1e-9, max_=max(y) + 1e-9)
    f.plot(list(range(len(y))), y)
    return f.show(legend=False)

def render(state: State) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=11),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    # Header
    title = Text("GEPA Optimization — Live Console", style="bold white")
    subtitle = Text(f"Source: {state.filename or 'stdin'}", style="dim")
    layout["header"].update(
        Panel(Group(title, subtitle), box=box.ROUNDED)
    )

    # Left: key metrics table
    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("Metric", style="cyan", no_wrap=True)
    tbl.add_column("Value", style="bold")

    last_best = state.best_scores[-1][1] if state.best_scores else float("nan")
    tbl.add_row("Best Score", f"{last_best:.4f}" if not math.isnan(last_best) else "—")
    tbl.add_row("Validation Acc", f"{state.val_acc:.4f}" if state.val_acc is not None else "—")
    tbl.add_row("Total Cost (USD)", f"${state.cost_usd:.4f}")
    elapsed = time.time() - state.start
    tbl.add_row("Elapsed", f"{elapsed:6.1f}s")
    tbl.add_row("Status", state.status)
    tbl.add_row("Trials", str(state.trial_count))

    layout["left"].update(
        Panel(tbl, title="Run Metrics", border_style="cyan")
    )

    # Right: optimization curve
    y = [v for _, v in state.best_scores]
    plot_txt = sparkline(y) if y else "Waiting for scores…"
    layout["right"].update(
        Panel(Text.from_ansi(plot_txt), title="Optimization Curve (best_score over time)", border_style="magenta")
    )

    # Footer: recent events
    ev_text = Text()
    if state.events:
        for e in list(state.events):
            ev_text.append(e + "\n")
    else:
        ev_text.append("No events yet…\n")
    layout["footer"].update(
        Panel(ev_text, title="Recent Events", border_style="green")
    )
    return layout

# ------------ Tail / ingest ------------

def follow_file(fp: io.TextIOBase):
    """Follow a file like tail -f."""
    fp.seek(0, os.SEEK_END)
    while True:
        line = fp.readline()
        if not line:
            time.sleep(0.2)
            continue
        yield line

def ingest_line(state: State, line: str):
    """Parse a line and update state."""
    # best_score sources
    m = BEST_SCORE_EQ_RE.search(line)
    if m:
        v = float(m.group(1))
        state.push_best(v)
        state.add_event(f"best_score updated → {v:.4f}")
    else:
        vj = try_json_num(line, "best_score")
        if vj is not None:
            state.push_best(vj)
            state.add_event(f"best_score (json) → {vj:.4f}")

    # trial results (for optimization curve)
    if "prompt.learning.trial.results" in line:
        mj = try_json_num(line, "mean")
        if mj is not None:
            state.trial_count += 1
            # Update best score if this trial is better
            if not state.best_scores or mj > state.best_scores[-1][1]:
                state.push_best(mj)
                state.add_event(f"trial {state.trial_count} → {mj:.4f}")

    # validation accuracy
    mv = VAL_ACC_RE.search(line)
    if mv:
        try:
            va = float(mv.group(1))
            state.val_acc = va
            state.add_event(f"val_acc → {va:.4f}")
        except ValueError:
            pass

    # cost
    mj = USD_RE.search(line)
    if mj:
        try:
            state.cost_usd = float(mj.group(1))
            state.add_event(f"total_usd → ${state.cost_usd:.4f}")
        except ValueError:
            pass

    mb = BILLED_RE.search(line)
    if mb:
        try:
            billed = float(mb.group(1))
            state.cost_usd = max(state.cost_usd, billed)
            state.add_event(f"billed → ${billed:.4f}")
        except ValueError:
            pass

    # status
    if "status=succeeded" in line or '"status":"succeeded"' in line:
        state.status = "succeeded"
        state.add_event("job succeeded")
    elif "failed" in line.lower() or '"status":"failed"' in line:
        state.status = "failed"
        state.add_event("job failed")

def main():
    ap = argparse.ArgumentParser(
        description="Live terminal dashboard for GEPA logs (Rich + Plotille)."
    )
    ap.add_argument("--log", type=str, help="Path to log file (else read stdin).")
    args = ap.parse_args()

    console = Console()
    state = State(max_events=12)
    state.filename = args.log

    def iter_lines():
        if args.log:
            with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
                for line in follow_file(f):
                    yield line
        else:
            for line in sys.stdin:
                yield line

    with Live(render(state), refresh_per_second=8, console=console, screen=True) as live:
        for ln in iter_lines():
            ingest_line(state, ln)
            # Update the live display
            live.update(render(state))
            # Throttle redraws a bit for heavy streams
            time.sleep(0.02)

if __name__ == "__main__":
    main()

