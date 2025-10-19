# Task App Config for Crafter RL: Dense Stepwise Rewards

Goal: Allow configuring the Crafter task app to enable/disable dense (decision-stepwise) event rewards and pass that choice from the RL config, through the backend, into the task app’s /rollout execution. This should be broader than just policy config – a `task_app_config` concept – but we can implement with the existing `env.config` today and optionally add a top-level alias later.

## Findings (current behaviour)

- Rollout request contract already supports two config payloads:
  - `env.config: dict` and `policy.config: dict`
- The hosted Crafter rollout implementation already supports decision-stepwise rewards, controlled via a `step_rewards` block in either `policy.config` or `env.config`.
- When active, it computes per-decision “unique achievement” deltas and attaches per-turn metadata; it also returns `decision_samples` when enabled.

Key locations and behaviour:

- Rollout schema (env/policy config):
```51:87:synth-ai/synth_ai/task/contracts.py
class RolloutEnvSpec(BaseModel):
    env_id: str | None = None
    env_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None

class RolloutPolicySpec(BaseModel):
    policy_id: str | None = None
    policy_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
```

- Crafter hosted rollout reads step-reward config from policy, then env; gates on `enabled` and `mode == "decision_stepwise"`:
```1041:1067:synth-ai/examples/warming_up_to_rl/task_app/synth_envs_hosted/rollout.py
# Stepwise reward configuration (Crafter shaping; gate on explicit enable)
step_rewards_cfg_raw: dict[str, Any] = {}
...
if not step_rewards_cfg_raw:
    if isinstance(request.env.config, dict):
        step_rewards_cfg_raw = dict(request.env.config.get("step_rewards") or {})

step_rewards_enabled = bool(step_rewards_cfg_raw.get("enabled", False))
step_rewards_mode = str(step_rewards_cfg_raw.get("mode") or "off").lower()
...
step_rewards_active = step_rewards_enabled and step_rewards_mode == "decision_stepwise"
```

- When active, it computes decision-level indicators and metadata, and adds to each step’s `info.meta.decision_rewards`; also accumulates `decision_samples`:
```1554:1596:synth-ai/examples/warming_up_to_rl/task_app/synth_envs_hosted/rollout.py
if step_rewards_active:
    decision_actions = _summarize_tool_calls(pending_tool_calls)
    stepwise_info, decision_record, stats = compute_stepwise_reward(
        prev_achievements or {},
        new_achievement_state,
        decision_index,
        decision_actions,
        step_rewards_indicator_lambda,
    )
    ...
    # Compute decision-level rewards (absolute vs unique) and attach to metadata
    turned_true = set(stepwise_info.get("new_achievements") or [])
    seen_before = set(episode_seen_achievements)
    new_unique = sorted(turned_true - seen_before)
    ach_delta = int(len(turned_true))
    unique_delta = int(len(new_unique))
    meta_block = (_info.get("meta") if isinstance(_info.get("meta"), dict) else {})
    decision_rewards = {"turn": int(decision_index), "ach_delta": ach_delta, "unique_delta": unique_delta, "all": all_list, "unique": new_unique}
    decision_rewards_meta = decision_rewards
    meta_block["decision_rewards"] = decision_rewards
    _info["meta"] = meta_block
    episode_seen_achievements.update(turned_true)
    decision_samples.append(decision_record)
```

- The simpler published Crafter app (`examples/warming_up_to_rl/task_app/grpo_crafter.py`) sets sane defaults for `step_rewards` in both env and policy when it aliases math → crafter, but the hosted rollout above is the one actually used in production paths.
```479:490:synth-ai/examples/warming_up_to_rl/task_app/grpo_crafter.py
env_cfg.setdefault("step_rewards", dict(DEFAULT_ALIAS_STEP_REWARDS))
...
policy_cfg.setdefault("step_rewards", dict(DEFAULT_ALIAS_STEP_REWARDS))
```

- Backend RPC: The backend constructs the rollout HTTP payload with both env_config and policy_config; these are forwarded to the task app `/rollout`:
```456:470:monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/evaluation/evaluator.py
payload = {
    "run_id": run_id,
    "env": {"env_name": env_name, "config": env_config, "seed": seed},
    "policy": {"policy_name": policy_name, "config": policy_config},
    "ops": ops,
    "record": {"trajectories": True, "logprobs": False, "value": False},
    "on_done": on_done,
}
```

- RL config ingestion: The CLI forwards the full TOML in the job payload. The backend trainer flattens some rollout options and (optionally) picks up `rollout.env_config`:
```364:393:monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/training/clustered_trainer.py
# Rollout config
if "rollout" in config_dict:
    flat_config["env_name"] = config_dict["rollout"].get("env_name", "crafter")
    ...
    if "env_config" in config_dict["rollout"]:
        flat_config["env_config"] = config_dict["rollout"]["env_config"]
```

Implication: We can carry a broader "task app config" today via `rollout.env_config` without changing wire contracts. The task app already reads `env.config.step_rewards`.

## Proposed configuration shape (TOML)

Recommended to use `rollout.env_config.step_rewards` so the backend passes it through to the task app:

```toml
[rollout]
env_name = "crafter"
policy_name = "crafter-react"
max_turns = 10
ops = ["agent", "env"]

[rollout.env_config.step_rewards]
# Toggle dense per-decision rewards
enabled = true
# Supported: "off" | "decision_stepwise" | (future) "env_sparse"
mode = "decision_stepwise"
# Reward = indicator_lambda * I(unique_achievements_delta > 0)
indicator_lambda = 1.0
# Reserved for shaped/intermediate signals (currently unused)
step_beta = 0.0
```

Optional (policy sampling, still supported via `policy.config` in the task app runner):

```toml
[rollout.policy_config]
temperature = 0.2
top_p = 0.95
max_tokens = 512
```

Notes:
- The hosted Crafter rollout checks `policy.config.step_rewards` first, then falls back to `env.config.step_rewards`. Prefer `env_config` as the canonical place for app-level settings.
- If you want the app to disable stepwise rewards entirely, set `enabled=false` or `mode="off"`.

## Wire and data flow

1) CLI → Backend: CLI includes the entire TOML in the job payload (`build_rl_payload`).
2) Backend → Trainer: Trainer flattens rollout properties and can include `env_config`.
3) Trainer → Task App: Rollout HTTP payload embeds `env.config` and `policy.config`.
4) Task App: Hosted rollout computes decision-level metadata and returns:
   - `RolloutStep.info.meta.decision_rewards` with `{turn, ach_delta, unique_delta, ...}`
   - `trajectory.decision_samples` summarizing per-turn reward inputs

## Minimal code changes to adopt

- synth-ai (optional):
  - Add example configs under `examples/warming_up_to_rl/configs/*.toml` using `[rollout.env_config.step_rewards]`.
  - Document this block in docs and the multi_step walkthrough.

- monorepo backend:
  - Verify trainer always passes `rollout.env_config` (ClusteredTrainerConfig appears to support it; ensure it flows into the runtime’s rollout request builder in the trainer where the payload is assembled).
  - No contract changes needed: task app already reads from `env.config`.

- Task App:
  - Already supports the block; no changes needed for the hosted Crafter rollout.
  - If you want a first-class `task_app_config` top-level, we can add an alias resolver that copies `config["task_app_config"]` → `env.config` inside the rollout executor.

## Open questions / follow-ups

- Does the current trainer consume `decision_samples` or `step.info.meta.decision_rewards` for credit assignment? If not, wire this into the per-step reward/advantage pipeline.
- Decide whether to disable the default enabling of stepwise rewards in `grpo_crafter.py` aliases (`DEFAULT_ALIAS_STEP_REWARDS`) so the TOML fully drives behaviour.
- Standardize on `env_config.step_rewards` for app-level settings across environments.

## Reference: CRAfter RL LoRA example (expected first 10 rewards)
These are the first ten batch rewards printed at RL start:
```
- INFO - All batch rewards: [0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.3125, 0.375, 0.4375, 0.5, 0.9375]
```
