import { createStackedLineGraph } from "./graph-utils"

export const RLEntropyKlGraph = createStackedLineGraph([
  {
    title: "entropy",
    metricNames: [
      "entropy",
      "policy_entropy",
      "rl.entropy",
    ],
    xLabel: "step",
    decimals: 4,
  },
  {
    title: "kl",
    metricNames: [
      "kl_divergence",
      "kl",
      "rl.kl",
      "policy_kl",
    ],
    xLabel: "step",
    decimals: 4,
  },
])
