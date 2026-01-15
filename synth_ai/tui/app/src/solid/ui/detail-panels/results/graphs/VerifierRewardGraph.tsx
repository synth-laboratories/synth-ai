import { createLineGraph } from "./graph-utils"

export const VerifierRewardGraph = createLineGraph({
  title: "verifier reward",
  metricNames: [
    "verifier.reward",
    "verifier_reward",
    "verifier.reward_mean",
    "verifier_reward_mean",
  ],
  xLabel: "step",
  decimals: 3,
})
