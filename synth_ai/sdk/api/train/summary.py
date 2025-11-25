"""Final summary display for prompt learning jobs."""

from __future__ import annotations

from typing import Any

import click

from .utils import http_get


def display_prompt_learning_summary(
    *,
    job_id: str,
    backend_base: str,
    api_key: str,
    optimization_curve: list[tuple[int, float]] | None = None,
    show_curve: bool = False,
    algorithm: str | None = None,
    log_writer: Any | None = None,
) -> None:
    """Display comprehensive final summary for prompt learning jobs.
    
    Args:
        log_writer: Optional callable that takes a string and writes to both console and log file.
                    If None, uses click.echo() only.
    """
    # Use log_writer if provided, otherwise use click.echo
    def write_output(text: str) -> None:
        if log_writer:
            log_writer(text)
        else:
            click.echo(text)
    
    # Fetch final job status and events
    try:
        # Fetch job status
        job_url = f"{backend_base}/prompt-learning/online/jobs/{job_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = http_get(job_url, headers=headers, timeout=30.0)
        
        if resp.status_code != 200:
            write_output(f"⚠️  Could not fetch job status (status={resp.status_code})")
            return
        
        resp.json()  # Validate response is JSON
        
        # Fetch events
        events_url = f"{backend_base}/prompt-learning/online/jobs/{job_id}/events?limit=1000"
        events_resp = http_get(events_url, headers=headers, timeout=30.0)
        
        if events_resp.status_code != 200:
            write_output(f"⚠️  Could not fetch events (status={events_resp.status_code})")
            all_events = []
        else:
            events_data = events_resp.json()
            if isinstance(events_data, list):
                all_events = events_data
            elif isinstance(events_data, dict):
                all_events = events_data.get("events", [])
            else:
                all_events = []
        
        # Extract metrics from events
        policy_cost_usd = None
        proposal_cost_usd = None
        total_cost_usd = None
        n_rollouts = None
        rollout_tokens_millions = None
        time_seconds = None
        
        # Extract MIPRO-specific metrics first (needed for billing.end fallback)
        mipro_budget_events = [e for e in all_events if e.get('type') == 'mipro.budget.summary']
        mipro_baseline_events = [e for e in all_events if e.get('type') == 'mipro.baseline.test']
        mipro_topk_events = [e for e in all_events if e.get('type') == 'mipro.topk.evaluated']
        mipro_completed_events = [e for e in all_events if e.get('type') == 'mipro.job.completed']
        
        # Extract from billing.end event (works for both GEPA and MIPRO)
        billing_end_events = [e for e in all_events if e.get('type') == 'prompt.learning.billing.end']
        if billing_end_events:
            billing_data = billing_end_events[-1].get('data', {})
            time_seconds = billing_data.get('seconds')
            total_cost_usd = billing_data.get('total_usd')
            # For MIPRO, also extract costs from billing.end if category_costs not available
            if algorithm == "mipro" and not mipro_budget_events:
                # Fallback: try to extract from billing.end if available
                tokens_usd = billing_data.get('tokens_usd', 0.0) or 0.0
                # Estimate policy/proposal split (conservative: assume policy is 80% of tokens)
                if tokens_usd > 0:
                    policy_cost_usd = tokens_usd * 0.8
                    proposal_cost_usd = tokens_usd * 0.2
        
        # Extract from completed event
        completed_events = [e for e in all_events if e.get('type') == 'prompt.learning.completed']
        completed_data = {}
        if completed_events:
            completed_data = completed_events[-1].get('data', {})
            policy_cost_usd = completed_data.get('usd_tokens_rollouts', 0.0) or 0.0
            proposal_cost_usd = completed_data.get('usd_tokens_mutation', 0.0) or 0.0
            
            # Rollout tokens in millions
            rollouts_prompt = completed_data.get('rollouts_prompt_tokens', 0) or 0
            rollouts_completion = completed_data.get('rollouts_completion_tokens', 0) or 0
            rollouts_unknown = completed_data.get('rollouts_unknown_tokens', 0) or 0
            rollout_tokens_total = rollouts_prompt + rollouts_completion + rollouts_unknown
            rollout_tokens_millions = rollout_tokens_total / 1_000_000.0
        
        if algorithm == "mipro":
            # Try budget summary first, then fall back to job completed event, then billing.end
            if mipro_budget_events:
                budget_data = mipro_budget_events[-1].get('data', {})
                category_costs = budget_data.get('category_costs', {})
                if category_costs:
                    # Cost categories are "rollout" and "proposal" (lowercase)
                    policy_cost_usd = category_costs.get('rollout', 0.0) or category_costs.get('policy', 0.0) or 0.0
                    proposal_cost_usd = category_costs.get('proposal', 0.0) or 0.0
                
                # Extract token counts (policy tokens are rollout tokens)
                policy_tokens = budget_data.get('policy_tokens', 0) or 0
                proposer_tokens = budget_data.get('proposer_tokens', 0) or 0
                # Rollout tokens = policy tokens only (not proposer)
                rollout_tokens_millions = policy_tokens / 1_000_000.0
            elif mipro_completed_events:
                # Fallback to job completed event
                completed_data_mipro = mipro_completed_events[-1].get('data', {})
                category_costs_completed = completed_data_mipro.get('category_costs', {})
                if category_costs_completed:
                    policy_cost_usd = category_costs_completed.get('rollout', 0.0) or category_costs_completed.get('policy', 0.0) or 0.0
                    proposal_cost_usd = category_costs_completed.get('proposal', 0.0) or 0.0
                # Also try direct fields
                if not policy_cost_usd:
                    policy_cost_usd = completed_data_mipro.get('policy_cost_usd', 0.0) or 0.0
                if not proposal_cost_usd:
                    proposal_cost_usd = completed_data_mipro.get('proposal_cost_usd', 0.0) or 0.0
            # If total_cost_usd is still None, use it from billing.end
            if total_cost_usd is None and billing_end_events:
                total_cost_usd = billing_data.get('total_usd')
        
        # Extract rollout count from progress events
        progress_events = [e for e in all_events if e.get('type') == 'prompt.learning.progress']
        trial_rollouts = 0
        if progress_events:
            all_rollout_counts = [
                e.get('data', {}).get('rollouts_completed', 0) or 0
                for e in progress_events
                if e.get('data', {}).get('rollouts_completed') is not None
            ]
            if all_rollout_counts:
                trial_rollouts = max(all_rollout_counts)
        
        # For MIPRO, count rollouts from trial events
        if algorithm == "mipro":
            mipro_trial_events = [e for e in all_events if e.get('type') == 'mipro.trial.complete']
            mipro_fulleval_events = [e for e in all_events if e.get('type') == 'mipro.fulleval.complete']
            mipro_test_events = [e for e in all_events if e.get('type') == 'mipro.test.complete']
            
            trial_rollouts = 0
            for event in mipro_trial_events:
                data = event.get('data', {})
                num_seeds = data.get('num_seeds', 0) or 0
                trial_rollouts += num_seeds
            
            heldout_rollouts = 0
            for event in mipro_fulleval_events:
                data = event.get('data', {})
                num_seeds = data.get('num_seeds', 0) or 0
                heldout_rollouts += num_seeds
            
            for event in mipro_test_events:
                data = event.get('data', {})
                num_seeds = data.get('num_seeds', 0) or 0
                heldout_rollouts += num_seeds
            
            # Add baseline test rollouts
            if mipro_baseline_events:
                baseline_data = mipro_baseline_events[-1].get('data', {})
                baseline_seeds = baseline_data.get('seeds', [])
                if baseline_seeds:
                    heldout_rollouts += len(baseline_seeds)
            
            # Add top-k evaluation rollouts
            for event in mipro_topk_events:
                data = event.get('data', {})
                test_seeds = data.get('test_seeds', [])
                if test_seeds:
                    heldout_rollouts += len(test_seeds)
        else:
            # Add heldout evaluation rollouts (GEPA)
            heldout_rollouts = 0
            validation_summary_events = [e for e in all_events if e.get('type') == 'prompt.learning.validation.summary']
            if validation_summary_events:
                val_summary = validation_summary_events[-1].get('data', {})
                baseline = val_summary.get('baseline', {})
                results = val_summary.get('results', [])
                
                baseline_seeds = baseline.get('seeds', [])
                if baseline_seeds:
                    heldout_rollouts += len(baseline_seeds)
                
                for result in results:
                    result_seeds = result.get('seeds', [])
                    if result_seeds:
                        heldout_rollouts += len(result_seeds)
        
        n_rollouts = trial_rollouts + heldout_rollouts
        
        # Extract validation results
        validation_summary = None
        validation_summary_events = [e for e in all_events if e.get('type') == 'prompt.learning.validation.summary']
        if validation_summary_events:
            validation_summary = validation_summary_events[-1].get('data', {})
        
        # Build summary table
        write_output("\n" + "=" * 80)
        write_output("FINAL SUMMARY")
        write_output("=" * 80)
        
        rows = []
        
        # Costs
        cost_policy = f"${policy_cost_usd:.4f}" if policy_cost_usd is not None else "N/A"
        cost_proposal = f"${proposal_cost_usd:.4f}" if proposal_cost_usd is not None else "N/A"
        cost_total = f"${total_cost_usd:.4f}" if total_cost_usd is not None else "N/A"
        rows.append(("Cost", f"Policy: {cost_policy} | Proposal: {cost_proposal} | Total: {cost_total}"))
        
        # Rollouts
        rollouts_str = f"{n_rollouts}" if n_rollouts is not None else "N/A"
        if algorithm == "mipro" and mipro_budget_events:
            # For MIPRO, show policy (rollout) tokens separately from proposal tokens
            budget_data = mipro_budget_events[-1].get('data', {})
            policy_tokens = budget_data.get('policy_tokens', 0) or 0
            proposer_tokens = budget_data.get('proposer_tokens', 0) or 0
            policy_tokens_millions = policy_tokens / 1_000_000.0
            proposer_tokens_millions = proposer_tokens / 1_000_000.0
            tokens_str = f"Policy: {policy_tokens_millions:.4f}M | Proposal: {proposer_tokens_millions:.4f}M"
        else:
            tokens_str = f"{rollout_tokens_millions:.4f}M" if rollout_tokens_millions is not None else "N/A"
        rows.append(("Rollouts", f"N: {rollouts_str} | Tokens: {tokens_str}"))
        
        # Throughput (extract from completed event for both GEPA and MIPRO)
        rollouts_per_min = completed_data.get('rollouts_per_minute')
        tokens_per_min = completed_data.get('tokens_per_minute')
        # For MIPRO, also check billing.end if not in completed event
        if algorithm == "mipro" and (rollouts_per_min is None or tokens_per_min is None) and billing_end_events:
            billing_data = billing_end_events[-1].get('data', {})
            # Calculate from time_seconds and total_rollouts if available
            if time_seconds and time_seconds > 0 and rollouts_per_min is None and n_rollouts is not None:
                rollouts_per_min = (n_rollouts / time_seconds * 60.0) if n_rollouts > 0 else None
            if time_seconds and time_seconds > 0 and tokens_per_min is None and rollout_tokens_millions is not None:
                tokens_per_min = (rollout_tokens_millions * 1_000_000.0 / time_seconds * 60.0) if rollout_tokens_millions > 0 else None
        if rollouts_per_min is not None or tokens_per_min is not None:
            throughput_parts = []
            if rollouts_per_min is not None:
                throughput_parts.append(f"Rollouts: {rollouts_per_min:.1f}/min")
            if tokens_per_min is not None:
                tokens_per_min_millions = tokens_per_min / 1_000_000.0
                throughput_parts.append(f"Tokens: {tokens_per_min_millions:.4f}M/min")
            rows.append(("Throughput", " | ".join(throughput_parts)))
        
        # Time
        if time_seconds is not None:
            rows.append(("Time", f"{time_seconds:.1f}s"))
        
        # Finish Reason (if available)
        finish_reason = completed_data.get('finish_reason')
        if finish_reason:
            rows.append(("Finish Reason", finish_reason))
        
        # Heldout Evaluation
        if algorithm == "mipro":
            # Extract MIPRO baseline and top-k results
            baseline_score = None
            if mipro_baseline_events:
                baseline_data = mipro_baseline_events[-1].get('data', {})
                baseline_score = baseline_data.get('score') or baseline_data.get('test_score')
            
            # Show baseline first
            if baseline_score is not None:
                rows.append(("Baseline", f"Accuracy: {baseline_score:.4f}"))
            
            # Show all top-K candidates (sorted by rank)
            if mipro_topk_events:
                # Sort by rank to ensure correct order
                sorted_topk = sorted(mipro_topk_events, key=lambda e: e.get('data', {}).get('rank', 999))
                for event in sorted_topk:
                    data = event.get('data', {})
                    rank = data.get('rank', 0)
                    test_score = data.get('test_score')
                    if test_score is not None and baseline_score is not None:
                        delta = test_score - baseline_score
                        delta_pct = (delta / baseline_score * 100) if baseline_score > 0 else 0
                        rows.append((f"Candidate {rank}", f"Accuracy: {test_score:.4f} (Δ{delta:+.4f}, {delta_pct:+.1f}% vs baseline)"))
                    elif test_score is not None:
                        rows.append((f"Candidate {rank}", f"Accuracy: {test_score:.4f}"))
        elif validation_summary:
            baseline = validation_summary.get('baseline', {})
            results = validation_summary.get('results', [])
            baseline_acc = baseline.get('accuracy')
            
            if baseline_acc is not None and results and len(results) > 0:
                # Only show candidate 1 (not candidate 2)
                    result = results[0]
                    result_acc = result.get('accuracy')
                    if result_acc is not None:
                        delta = result_acc - baseline_acc
                        rows.append(("Candidate 1", f"Accuracy: {result_acc:.4f} (Δ{delta:+.4f} vs baseline)"))
        
        # Display table
        max_label_len = max(len(row[0]) for row in rows) if rows else 0
        for label, value in rows:
            write_output(f"{label:>{max_label_len}} {value}")
        
        write_output("=" * 80)
        
        # Display optimization curve if requested
        if show_curve and optimization_curve:
            from synth_ai.cli.lib.plotting import plot_optimization_curve
            
            trial_counts = [t for t, _ in optimization_curve]
            best_scores = [s for _, s in optimization_curve]
            curve_plot = plot_optimization_curve(
                trial_counts=trial_counts,
                best_scores=best_scores,
                title="Optimization Curve: Best Score vs Trial Count",
            )
            write_output("\n" + curve_plot)
    
    except Exception as e:
        write_output(f"⚠️  Error displaying summary: {e}")


def _generate_summary_text(
    *,
    events: list[dict[str, Any]],
    algorithm: str | None = None,
    optimization_curve: list[tuple[int, float]] | None = None,
    backend_base: str | None = None,
    api_key: str | None = None,
    job_id: str | None = None,
) -> tuple[str, str]:
    """Generate summary table and curve text from events (for file output).
    
    Returns:
        Tuple of (summary_table_text, curve_text)
    """
    # Reuse the same extraction logic as display_prompt_learning_summary
    # This is a simplified version that builds text directly
    
    # Extract metrics (same logic as display_prompt_learning_summary)
    policy_cost_usd = None
    proposal_cost_usd = None
    total_cost_usd = None
    n_rollouts = None
    rollout_tokens_millions = None
    time_seconds = None
    
    # Extract MIPRO-specific metrics first
    mipro_budget_events = [e for e in events if e.get('type') == 'mipro.budget.summary']
    mipro_baseline_events = [e for e in events if e.get('type') == 'mipro.baseline.test']
    mipro_topk_events = [e for e in events if e.get('type') == 'mipro.topk.evaluated']
    # mipro_completed_events not currently used but may be needed for future metrics
    
    # Extract from billing.end event
    billing_end_events = [e for e in events if e.get('type') == 'prompt.learning.billing.end']
    if billing_end_events:
        billing_data = billing_end_events[-1].get('data', {})
        time_seconds = billing_data.get('seconds')
        total_cost_usd = billing_data.get('total_usd')
    
    # Extract from completed event
    completed_events = [e for e in events if e.get('type') == 'prompt.learning.completed']
    completed_data = {}
    if completed_events:
        completed_data = completed_events[-1].get('data', {})
        policy_cost_usd = completed_data.get('usd_tokens_rollouts', 0.0) or 0.0
        proposal_cost_usd = completed_data.get('usd_tokens_mutation', 0.0) or 0.0
        
        rollouts_prompt = completed_data.get('rollouts_prompt_tokens', 0) or 0
        rollouts_completion = completed_data.get('rollouts_completion_tokens', 0) or 0
        rollouts_unknown = completed_data.get('rollouts_unknown_tokens', 0) or 0
        rollout_tokens_total = rollouts_prompt + rollouts_completion + rollouts_unknown
        rollout_tokens_millions = rollout_tokens_total / 1_000_000.0
    
    if algorithm == "mipro" and mipro_budget_events:
            budget_data = mipro_budget_events[-1].get('data', {})
            category_costs = budget_data.get('category_costs', {})
            if category_costs:
                policy_cost_usd = category_costs.get('rollout', 0.0) or category_costs.get('policy', 0.0) or 0.0
                proposal_cost_usd = category_costs.get('proposal', 0.0) or 0.0
            
            policy_tokens = budget_data.get('policy_tokens', 0) or 0
            proposer_tokens = budget_data.get('proposer_tokens', 0) or 0
            rollout_tokens_millions = policy_tokens / 1_000_000.0
    
    # Extract rollout count
    if algorithm == "mipro":
        mipro_trial_events = [e for e in events if e.get('type') == 'mipro.trial.complete']
        mipro_fulleval_events = [e for e in events if e.get('type') == 'mipro.fulleval.complete']
        mipro_test_events = [e for e in events if e.get('type') == 'mipro.test.complete']
        
        trial_rollouts = 0
        for event in mipro_trial_events:
            data = event.get('data', {})
            num_seeds = data.get('num_seeds', 0) or 0
            trial_rollouts += num_seeds
        
        heldout_rollouts = 0
        for event in mipro_fulleval_events:
            data = event.get('data', {})
            num_seeds = data.get('num_seeds', 0) or 0
            heldout_rollouts += num_seeds
        
        for event in mipro_test_events:
            data = event.get('data', {})
            num_seeds = data.get('num_seeds', 0) or 0
            heldout_rollouts += num_seeds
        
        if mipro_baseline_events:
            baseline_data = mipro_baseline_events[-1].get('data', {})
            baseline_seeds = baseline_data.get('seeds', [])
            if baseline_seeds:
                heldout_rollouts += len(baseline_seeds)
        
        for event in mipro_topk_events:
            data = event.get('data', {})
            test_seeds = data.get('test_seeds', [])
            if test_seeds:
                heldout_rollouts += len(test_seeds)
        
        n_rollouts = trial_rollouts + heldout_rollouts
    
    # Build summary table text
    summary_lines = []
    rows = []
    
    # Costs
    cost_policy = f"${policy_cost_usd:.4f}" if policy_cost_usd is not None else "N/A"
    cost_proposal = f"${proposal_cost_usd:.4f}" if proposal_cost_usd is not None else "N/A"
    cost_total = f"${total_cost_usd:.4f}" if total_cost_usd is not None else "N/A"
    rows.append(("Cost", f"Policy: {cost_policy} | Proposal: {cost_proposal} | Total: {cost_total}"))
    
    # Rollouts
    rollouts_str = f"{n_rollouts}" if n_rollouts is not None else "N/A"
    if algorithm == "mipro" and mipro_budget_events:
        budget_data = mipro_budget_events[-1].get('data', {})
        policy_tokens = budget_data.get('policy_tokens', 0) or 0
        proposer_tokens = budget_data.get('proposer_tokens', 0) or 0
        policy_tokens_millions = policy_tokens / 1_000_000.0
        proposer_tokens_millions = proposer_tokens / 1_000_000.0
        tokens_str = f"Policy: {policy_tokens_millions:.4f}M | Proposal: {proposer_tokens_millions:.4f}M"
    else:
        tokens_str = f"{rollout_tokens_millions:.4f}M" if rollout_tokens_millions is not None else "N/A"
    rows.append(("Rollouts", f"N: {rollouts_str} | Tokens: {tokens_str}"))
    
    # Throughput
    rollouts_per_min = completed_data.get('rollouts_per_minute')
    tokens_per_min = completed_data.get('tokens_per_minute')
    if rollouts_per_min is not None or tokens_per_min is not None:
        throughput_parts = []
        if rollouts_per_min is not None:
            throughput_parts.append(f"Rollouts: {rollouts_per_min:.1f}/min")
        if tokens_per_min is not None:
            tokens_per_min_millions = tokens_per_min / 1_000_000.0
            throughput_parts.append(f"Tokens: {tokens_per_min_millions:.4f}M/min")
        rows.append(("Throughput", " | ".join(throughput_parts)))
    
    # Time
    if time_seconds is not None:
        rows.append(("Time", f"{time_seconds:.1f}s"))
    
    # Finish Reason (if available)
    completed_events_for_text = [e for e in events if e.get('type') == 'prompt.learning.completed']
    if completed_events_for_text:
        completed_data_for_text = completed_events_for_text[-1].get('data', {})
        finish_reason = completed_data_for_text.get('finish_reason')
        if finish_reason:
            rows.append(("Finish Reason", finish_reason))
    
    # Heldout Evaluation
    if algorithm == "mipro":
        baseline_score = None
        if mipro_baseline_events:
            baseline_data = mipro_baseline_events[-1].get('data', {})
            baseline_score = baseline_data.get('score') or baseline_data.get('test_score')
        
        if baseline_score is not None:
            rows.append(("Baseline", f"Accuracy: {baseline_score:.4f}"))
        
        if mipro_topk_events:
            sorted_topk = sorted(mipro_topk_events, key=lambda e: e.get('data', {}).get('rank', 999))
            for event in sorted_topk:
                data = event.get('data', {})
                rank = data.get('rank', 0)
                test_score = data.get('test_score')
                if test_score is not None and baseline_score is not None:
                    delta = test_score - baseline_score
                    delta_pct = (delta / baseline_score * 100) if baseline_score > 0 else 0
                    rows.append((f"Candidate {rank}", f"Accuracy: {test_score:.4f} (Δ{delta:+.4f}, {delta_pct:+.1f}% vs baseline)"))
                elif test_score is not None:
                    rows.append((f"Candidate {rank}", f"Accuracy: {test_score:.4f}"))
    
    # Format table
    max_label_len = max(len(row[0]) for row in rows) if rows else 0
    for label, value in rows:
        summary_lines.append(f"{label:>{max_label_len}} {value}")
    
    summary_text = "\n".join(summary_lines)
    
    # Generate curve text
    curve_text = ""
    if optimization_curve:
        from synth_ai.cli.lib.plotting import plot_optimization_curve
        trial_counts = [t for t, _ in optimization_curve]
        best_scores = [s for _, s in optimization_curve]
        curve_text = plot_optimization_curve(
            trial_counts=trial_counts,
            best_scores=best_scores,
            title="Optimization Curve: Best Score vs Trial Count",
        )
    
    return summary_text, curve_text

