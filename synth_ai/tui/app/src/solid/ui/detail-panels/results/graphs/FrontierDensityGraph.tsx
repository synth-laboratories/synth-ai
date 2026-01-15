import { createLineGraph } from "./graph-utils"

export const FrontierDensityGraph = createLineGraph({
  title: "frontier density",
  metricNames: [
    "gepa.frontier.density",
    "frontier.density",
    "frontier_density",
  ],
  xLabel: "step",
  decimals: 3,
})
