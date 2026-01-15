import { createHistogramGraph } from "./graph-utils"

export const RewardDistributionGraph = createHistogramGraph({
  title: "reward distribution",
  metricNames: [
    "gepa.transformation.mean_reward",
    "mipro.minibatch_*",
    "mipro.full_*",
    "rl.reward",
    "epoch.reward_mean",
    "eval.reward_mean",
    "reward",
    "train.reward",
    "eval.reward",
    "policy.reward",
    "verifier.reward",
    "local_api_reward",
    "verifier_reward",
    "outcome_reward",
    "fused_reward",
    "reward_mean",
    "reward_avg",
    "reward_p50",
    "reward_p90",
  ],
  bins: 8,
  decimals: 3,
})
