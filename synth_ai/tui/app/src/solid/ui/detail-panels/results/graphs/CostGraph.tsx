import { createLineGraph } from "./graph-utils"

export const CostGraph = createLineGraph({
  title: "cost (usd)",
  metricNames: [
    "total_cost_usd",
    "cost_usd",
    "cost",
    "cost_total_usd",
    "usd_spend",
  ],
  xLabel: "step",
  decimals: 4,
})
