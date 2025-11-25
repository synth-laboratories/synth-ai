"""Utilities for parsing job artifacts into structured summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def _load_latest_json(directory: Path, pattern: str) -> tuple[dict[str, Any], Path] | tuple[None, None]:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8") as fh:
                return json.load(fh), candidate
        except Exception:
            continue
    return None, None


@dataclass(slots=True)
class LearningCurvePoint:
    rollout_count: int | None = None
    performance: float | None = None
    checkpoint_pct: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_count": self.rollout_count,
            "performance": self.performance,
            "checkpoint_pct": self.checkpoint_pct,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ResultSummary:
    """Structured representation of job outputs used for DB + CLI."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    learning_curve_points: list[LearningCurvePoint] = field(default_factory=list)
    best_score: float | None = None
    baseline_score: float | None = None
    total_rollouts: int | None = None
    total_time: float | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stats": self.stats,
            "learning_curve": [point.to_dict() for point in self.learning_curve_points],
            "best_score": self.best_score,
            "baseline_score": self.baseline_score,
            "total_rollouts": self.total_rollouts,
            "total_time": self.total_time,
            "artifacts": self.artifacts,
        }


def _parse_learning_curve(data: dict[str, Any]) -> list[LearningCurvePoint]:
    curve = data.get("curve") or data.get("checkpoints") or []
    points: list[LearningCurvePoint] = []
    if isinstance(curve, Iterable):
        for entry in curve:
            if not isinstance(entry, dict):
                continue
            points.append(
                LearningCurvePoint(
                    rollout_count=entry.get("rollout_count") or entry.get("rollouts"),
                    performance=entry.get("performance")
                    or entry.get("score")
                    or entry.get("aggregate_score"),
                    checkpoint_pct=entry.get("checkpoint_pct"),
                    metadata={k: v for k, v in entry.items() if k not in {"rollout_count", "performance", "checkpoint_pct"}},
                )
            )
    return points


def _extract_rollouts_from_output(stdout: str, stderr: str, results_folder: Path | None = None) -> int | None:
    """Extract rollout count from stdout/stderr and log files by looking for rollout patterns."""
    import re
    
    # Look for patterns like "[BANKING77_ROLLOUT] ... index=75" or "rollout.*index=(\d+)"
    patterns = [
        r'\[.*ROLLOUT\].*index=(\d+)',
        r'rollout.*index=(\d+)',
        r'rollout\s+(\d+)',
        r'completed\s+(\d+)\s+rollouts?',
        r'total.*rollouts?[:\s]+(\d+)',
        r'"count":\s*(\d+).*rollout',  # JSON stats: "count": 59 in rollout_duration_stats
        r'rollout.*"count":\s*(\d+)',  # JSON stats in rollout context
    ]
    
    max_rollout = None
    texts_to_check = [stdout, stderr]
    
    # Also check log files if results_folder is provided
    if results_folder and results_folder.exists():
        log_files = sorted(
            results_folder.glob("*_log_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Found {len(log_files)} log files in {results_folder}")
        for log_file in log_files[:2]:  # Check up to 2 most recent log files
            try:
                log_content = log_file.read_text(encoding="utf-8", errors="ignore")
                texts_to_check.append(log_content)
                logger.debug(f"Read log file: {log_file.name} ({len(log_content)} chars)")
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
    
    for text in texts_to_check:
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                rollout_nums = [int(m) for m in matches if m.isdigit()]
                if rollout_nums:
                    max_rollout = max(max_rollout or 0, max(rollout_nums))
    
    # Also try to extract from JSON stats if present
    if max_rollout is None:
        for text in texts_to_check:
            # Look for "rollout_duration_stats" with JSON containing "count"
            # Format in file: rollout_duration_stats": "{\n  \"count\": 30,\n  ...}"
            # When read, escaped quotes stay as \", so we need to match \"count\"
            # Try pattern with escaped quotes first (most common case)
            stats_match = re.search(
                r'rollout_duration_stats.*?\\"count\\"\s*:\s*(\d+)',
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if stats_match:
                count = int(stats_match.group(1))
                max_rollout = max(max_rollout or 0, count) if max_rollout else count
            else:
                # Fallback: Try unescaped quotes (in case file was processed)
                stats_match2 = re.search(
                    r'rollout_duration_stats.*?"count"\s*:\s*(\d+)',
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if stats_match2:
                    count = int(stats_match2.group(1))
                    max_rollout = max(max_rollout or 0, count) if max_rollout else count
                else:
                    # Last resort: Get all \"count\": N or "count": N values and take the max reasonable one
                    all_counts_escaped = re.findall(r'\\"count\\"\s*:\s*(\d+)', text, re.IGNORECASE)
                    all_counts_normal = re.findall(r'"count"\s*:\s*(\d+)', text, re.IGNORECASE)
                    all_counts = all_counts_escaped + all_counts_normal
                    if all_counts:
                        counts = [int(c) for c in all_counts if c.isdigit()]
                        if counts:
                            count = max(counts)
                            # Only use if it's a reasonable rollout count (not 0 or 1)
                            if count > 1:
                                max_rollout = max(max_rollout or 0, count) if max_rollout else count
    
    return max_rollout if max_rollout is not None else None


def _parse_text_results_file(results_file: Path) -> dict[str, Any]:
    """Parse GEPA/MIPRO text result files to extract scores and metadata."""
    import re
    
    result = {}
    try:
        content = results_file.read_text(encoding="utf-8")
        
        # Extract baseline score
        baseline_match = re.search(r'Baseline Score:\s*([\d.]+)', content, re.IGNORECASE)
        if baseline_match:
            result["baseline_score"] = float(baseline_match.group(1))
        
        # Extract best score
        best_match = re.search(r'Best Score:\s*([\d.]+)', content, re.IGNORECASE)
        if best_match:
            result["best_score"] = float(best_match.group(1))
        
        # Extract job ID
        job_id_match = re.search(r'Job ID:\s*(\S+)', content, re.IGNORECASE)
        if job_id_match:
            result["job_id"] = job_id_match.group(1)
            
    except Exception:
        pass
    
    return result


def collect_result_summary(results_folder: Path, stdout: str = "", stderr: str = "") -> ResultSummary:
    """Introspect result artifacts saved by prompt learning jobs.
    
    Args:
        results_folder: Path to results directory
        stdout: Standard output from job execution
        stderr: Standard error from job execution
        
    Returns:
        ResultSummary with parsed results
        
    Raises:
        AssertionError: If inputs are invalid
    """
    from .validation import validate_path
    
    # Validate inputs
    assert results_folder is not None, "results_folder cannot be None"
    path = validate_path(results_folder, "results_folder", must_exist=False)
    assert isinstance(stdout, str), (
        f"stdout must be str, got {type(stdout).__name__}"
    )
    assert isinstance(stderr, str), (
        f"stderr must be str, got {type(stderr).__name__}"
    )
    
    summary = ResultSummary()
    summary.stdout = stdout
    summary.stderr = stderr
    
    if not path.exists():
        # Try to extract rollouts from stdout/stderr even if no results folder
        extracted = _extract_rollouts_from_output(stdout, stderr, results_folder=None)
        if extracted is not None:
            assert extracted > 0, f"total_rollouts must be > 0, got {extracted}"
        summary.total_rollouts = extracted
        return summary

    learning_curve_data, curve_path = _load_latest_json(path, "*learning_curve*.json")
    if learning_curve_data:
        assert isinstance(learning_curve_data, dict), (
            f"learning_curve_data must be dict, got {type(learning_curve_data).__name__}"
        )
        summary.learning_curve_points = _parse_learning_curve(learning_curve_data)
        assert isinstance(summary.learning_curve_points, list), (
            f"learning_curve_points must be list, got {type(summary.learning_curve_points).__name__}"
        )
        
        points = [p for p in summary.learning_curve_points if p.performance is not None]
        if points:
            baseline = points[0].performance
            assert baseline is not None and 0 <= baseline <= 1, (
                f"baseline_score must be in [0, 1], got {baseline}"
            )
            summary.baseline_score = baseline
            
            best = max((p.performance for p in points if p.performance is not None), default=None)
            if best is not None:
                assert 0 <= best <= 1, f"best_score must be in [0, 1], got {best}"
            summary.best_score = best
            
            if points[-1].rollout_count is not None:
                assert points[-1].rollout_count > 0, (
                    f"rollout_count must be > 0, got {points[-1].rollout_count}"
                )
                summary.total_rollouts = points[-1].rollout_count
            elif learning_curve_data.get("total_rollouts") is not None:
                total = learning_curve_data.get("total_rollouts")
                assert isinstance(total, int) and total > 0, (
                    f"total_rollouts must be int > 0, got {total}"
                )
                summary.total_rollouts = total
        if curve_path:
            assert isinstance(curve_path, Path), (
                f"curve_path must be Path, got {type(curve_path).__name__}"
            )
            summary.artifacts["learning_curve_path"] = str(curve_path)

    stats_data, stats_path = _load_latest_json(path, "*stats*.json")
    if stats_data:
        assert isinstance(stats_data, dict), (
            f"stats_data must be dict, got {type(stats_data).__name__}"
        )
        summary.stats = stats_data
        total_time = stats_data.get("total_time", summary.total_time)
        if total_time is not None:
            assert isinstance(total_time, int | float) and total_time >= 0, (
                f"total_time must be >= 0, got {total_time}"
            )
        summary.total_time = total_time
        
        total_rollouts_from_stats = stats_data.get("total_rollouts")
        if total_rollouts_from_stats is not None:
            assert isinstance(total_rollouts_from_stats, int) and total_rollouts_from_stats > 0, (
                f"total_rollouts must be int > 0, got {total_rollouts_from_stats}"
            )
        summary.total_rollouts = summary.total_rollouts or total_rollouts_from_stats
        
        if stats_path:
            assert isinstance(stats_path, Path), (
                f"stats_path must be Path, got {type(stats_path).__name__}"
            )
            summary.artifacts["stats_path"] = str(stats_path)

    # If no JSON files found, try parsing text result files or extracting from stdout/stderr/log files
    if summary.total_rollouts is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "No rollouts found in JSON files, attempting extraction from stdout/stderr/log files. "
            "Results folder: %s",
            path,
        )
        extracted_rollouts = _extract_rollouts_from_output(stdout, stderr, results_folder=path)
        if extracted_rollouts is not None:
            assert extracted_rollouts > 0, (
                f"extracted_rollouts must be > 0, got {extracted_rollouts}"
            )
            summary.total_rollouts = extracted_rollouts
            logger.info(
                "✅ Extracted total_rollouts=%d from stdout/stderr/log files",
                extracted_rollouts,
            )
        else:
            logger.warning(
                "❌ Failed to extract rollouts from stdout/stderr/log files. "
                "Stdout length: %d, Stderr length: %d, Results folder exists: %s",
                len(stdout),
                len(stderr),
                path.exists() if path else False,
            )
    
    if summary.best_score is None or summary.baseline_score is None:
        # Look for text result files (gepa_results_*.txt, mipro_results_*.txt)
        result_files = sorted(
            path.glob("*_results_*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if result_files:
            text_results = _parse_text_results_file(result_files[0])
            assert isinstance(text_results, dict), (
                f"_parse_text_results_file must return dict, got {type(text_results).__name__}"
            )
            
            best_score = text_results.get("best_score")
            if best_score is not None:
                assert isinstance(best_score, int | float) and 0 <= best_score <= 1, (
                    f"best_score must be in [0, 1], got {best_score}"
                )
                summary.best_score = float(best_score)
            
            baseline_score = text_results.get("baseline_score")
            if baseline_score is not None:
                assert isinstance(baseline_score, int | float) and 0 <= baseline_score <= 1, (
                    f"baseline_score must be in [0, 1], got {baseline_score}"
                )
                summary.baseline_score = float(baseline_score)
            
            if result_files[0]:
                summary.artifacts["results_txt_path"] = str(result_files[0])

    best_prompt_candidates = sorted(
        path.glob("*best_prompt*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    best_prompt_file = next(iter(best_prompt_candidates), None)
    if best_prompt_file:
        summary.artifacts["best_prompt_path"] = str(best_prompt_file)

    return summary
