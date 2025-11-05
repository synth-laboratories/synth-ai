from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

import click

from ..client import StatusAPIClient
from ..errors import StatusAPIError
from ..formatters import console
from ..utils import common_options, resolve_context_config


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # Python 3.11 handles 'YYYY-mm-ddTHH:MM:SS.ssssss+00:00' and '...Z'
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _extract_total_usd(events: list[dict[str, Any]]) -> tuple[float, int]:
    """Return (usd_total, tokens_total) for an arbitrary job's events.

    Strategy:
    - Prefer a consolidated total from any *.completed event with total_usd
    - Next, prefer any *.billing.end event with total_usd
    - Otherwise, combine usage.recorded's usd_tokens with billing.sandboxes' usd
      and sum token counts if present
    Works for prompt learning and other job types that follow similar conventions.
    """
    total_usd = 0.0
    token_count = 0

    # Prefer consolidated totals from completion events (any namespace)
    for e in reversed(events):
        typ = str(e.get("type") or "").lower()
        if typ.endswith(".completed"):
            data = e.get("data") or {}
            try:
                total_usd = float(data.get("total_usd") or 0.0)
            except Exception:
                total_usd = 0.0
            # Try common token fields
            tc = 0
            for k in ("token_count_total", "token_count"):
                try:
                    tc = int(data.get(k) or 0)
                    if tc:
                        break
                except Exception:
                    pass
            if not tc:
                try:
                    tc = int((data.get("token_count_rollouts") or 0) + (data.get("token_count_mutation") or 0))
                except Exception:
                    tc = 0
            token_count = tc
            return total_usd, token_count

    # Next, billing.end if present with total_usd
    for e in reversed(events):
        typ = str(e.get("type") or "").lower()
        if typ.endswith("billing.end"):
            data = e.get("data") or {}
            try:
                total_usd = float(data.get("total_usd") or 0.0)
            except Exception:
                total_usd = 0.0
            # token_count may not be present here; fall through to usage tokens calc
            break

    # Fallback: combine usage + sandboxes (prompt learning style); generic scan
    usd_tokens = 0.0
    sandbox_usd = 0.0
    # token fields observed across tasks
    token_fields = ("token_count_total", "token_count", "tokens_in", "tokens_out",
                    "token_count_rollouts", "token_count_mutation")
    for e in events:
        typ = str(e.get("type") or "").lower()
        data = e.get("data") or {}
        # generic usage-style aggregation
        if "usage" in typ or typ.endswith("usage.recorded"):
            with contextlib.suppress(Exception):
                usd_tokens = float(data.get("usd_tokens") or data.get("usd_estimate") or 0.0)
            # accumulate tokens if any
            for k in token_fields:
                with contextlib.suppress(Exception):
                    token_count += int(data.get(k) or 0)
        # sandbox billing
        if typ.endswith("billing.sandboxes"):
            with contextlib.suppress(Exception):
                sandbox_usd += float(data.get("usd") or 0.0)
    return (total_usd or (usd_tokens + sandbox_usd)), token_count


@click.command("usage", help="Show recent usage (daily/weekly/monthly) and remaining budget if provided.")
@common_options()
@click.option("--budget-usd", type=float, default=None, help="Optional credit/budget to compute remaining.")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def usage_command(
    ctx: click.Context,
    base_url: str | None,
    api_key: str | None,
    timeout: float,
    budget_usd: float | None,
    output_json: bool,
) -> None:
    cfg = resolve_context_config(ctx, base_url=base_url, api_key=api_key, timeout=timeout)
    now = datetime.now(UTC)
    daily_cutoff = (now - timedelta(days=1)).isoformat()
    weekly_cutoff = (now - timedelta(days=7)).isoformat()
    monthly_cutoff = (now - timedelta(days=30)).isoformat()

    async def _run() -> tuple[dict[str, float | int], dict[str, float | int], dict[str, float | int]]:
        daily = {"usd": 0.0, "tokens": 0, "sandbox_seconds": 0.0}
        weekly = {"usd": 0.0, "tokens": 0, "sandbox_seconds": 0.0}
        monthly = {"usd": 0.0, "tokens": 0, "sandbox_seconds": 0.0}
        async with StatusAPIClient(cfg) as client:
            try:
                jobs = await client.list_jobs(created_after=weekly_cutoff)
            except StatusAPIError as exc:
                raise click.ClickException(f"Backend error: {exc}") from exc
            for j in jobs or []:
                job_id = str(j.get("job_id") or j.get("id") or "")
                if not job_id:
                    continue
                try:
                    events = await client.get_job_events(job_id, since=weekly_cutoff)
                except StatusAPIError:
                    events = []
                if not events:
                    continue
                # Use event timestamps for windowing
                # Weekly
                weekly_ev = [e for e in events if (_parse_iso(e.get("created_at")) or now) >= datetime.fromisoformat(weekly_cutoff)]
                w_usd, w_tok = _extract_total_usd(weekly_ev)
                weekly["usd"] += w_usd
                weekly["tokens"] += w_tok
                # sandbox seconds
                for e in weekly_ev:
                    if str(e.get("type") or "").lower().endswith("billing.sandboxes"):
                        with contextlib.suppress(Exception):
                            weekly["sandbox_seconds"] += float((e.get("data") or {}).get("seconds") or 0.0)
                # Daily
                daily_ev = [e for e in events if (_parse_iso(e.get("created_at")) or now) >= datetime.fromisoformat(daily_cutoff)]
                d_usd, d_tok = _extract_total_usd(daily_ev)
                daily["usd"] += d_usd
                daily["tokens"] += d_tok
                for e in daily_ev:
                    if str(e.get("type") or "").lower().endswith("billing.sandboxes"):
                        with contextlib.suppress(Exception):
                            daily["sandbox_seconds"] += float((e.get("data") or {}).get("seconds") or 0.0)
                # Monthly
                monthly_ev = [e for e in events if (_parse_iso(e.get("created_at")) or now) >= datetime.fromisoformat(monthly_cutoff)]
                m_usd, m_tok = _extract_total_usd(monthly_ev)
                monthly["usd"] += m_usd
                monthly["tokens"] += m_tok
                for e in monthly_ev:
                    if str(e.get("type") or "").lower().endswith("billing.sandboxes"):
                        with contextlib.suppress(Exception):
                            monthly["sandbox_seconds"] += float((e.get("data") or {}).get("seconds") or 0.0)
        return daily, weekly, monthly

    daily, weekly, monthly = __import__("asyncio").run(_run())

    if output_json:
        import json as _json
        payload: dict[str, Any] = {
            "daily": {
                "usd": round(float(daily["usd"]), 4),
                "tokens": int(daily["tokens"]),
                "sandbox_hours": round(float(daily["sandbox_seconds"]) / 3600.0, 4),
            },
            "weekly": {
                "usd": round(float(weekly["usd"]), 4),
                "tokens": int(weekly["tokens"]),
                "sandbox_hours": round(float(weekly["sandbox_seconds"]) / 3600.0, 4),
            },
            "monthly": {
                "usd": round(float(monthly["usd"]), 4),
                "tokens": int(monthly["tokens"]),
                "sandbox_hours": round(float(monthly["sandbox_seconds"]) / 3600.0, 4),
            },
        }
        if budget_usd is not None:
            payload["remaining_vs_budget"] = round(max(0.0, float(budget_usd) - float(weekly["usd"])), 4)
        console.print(_json.dumps(payload))
        return

    console.print(f"Daily usage: ${float(daily['usd']):.2f} | tokens {int(daily['tokens'])} | sandbox {float(daily['sandbox_seconds'])/3600.0:.2f}h")
    console.print(f"Weekly usage: ${float(weekly['usd']):.2f} | tokens {int(weekly['tokens'])} | sandbox {float(weekly['sandbox_seconds'])/3600.0:.2f}h")
    console.print(f"Monthly usage: ${float(monthly['usd']):.2f} | tokens {int(monthly['tokens'])} | sandbox {float(monthly['sandbox_seconds'])/3600.0:.2f}h")
    if budget_usd is not None:
        remaining = max(0.0, float(budget_usd) - float(weekly["usd"]))
        console.print(f"Remaining (vs weekly budget ${float(budget_usd):.2f}): ${remaining:.2f}")

