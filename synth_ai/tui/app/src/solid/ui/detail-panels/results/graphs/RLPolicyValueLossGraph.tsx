import { createStackedLineGraph } from "./graph-utils"

export const RLPolicyValueLossGraph = createStackedLineGraph([
  {
    title: "policy loss",
    metricNames: [
      "policy_loss",
      "rl.policy_loss",
      "loss.policy",
      "actor_loss",
    ],
    xLabel: "step",
    decimals: 4,
  },
  {
    title: "value loss",
    metricNames: [
      "value_loss",
      "rl.value_loss",
      "loss.value",
      "critic_loss",
    ],
    xLabel: "step",
    decimals: 4,
  },
])
