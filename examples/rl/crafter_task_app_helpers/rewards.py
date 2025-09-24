from __future__ import annotations

from typing import List, Dict, Any


def compute_decision_rewards(
    *,
    decision_summaries: List[Dict[str, Any]],
    total_achievements: int,
    step_beta: float,
    indicator_lambda: float,
) -> List[Dict[str, Any]]:
    """
    Given per-decision summaries and final achievements count, compute r_i for each decision:
        r_i = (T - i) * step_beta * A_T + indicator_lambda * indicator_i

    Returns a list of dicts with keys:
        {decision_index, reward, indicator_i, achievements_count, total_steps}
    """
    if not isinstance(decision_summaries, list):
        return []
    T = len(decision_summaries)
    out: List[Dict[str, Any]] = []
    for i, dec in enumerate(decision_summaries, start=1):
        indicator_i = 1 if bool(dec.get("indicator_i")) else 0
        r_i = (T - i) * float(step_beta) * float(total_achievements) + float(indicator_lambda) * float(indicator_i)
        out.append(
            {
                "decision_index": i,
                "reward": r_i,
                "indicator_i": indicator_i,
                "achievements_count": int(total_achievements),
                "total_steps": int(T),
            }
        )
    return out


