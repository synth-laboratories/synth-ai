import { createLineGraph } from "./graph-utils"

export const ErrorRateGraph = createLineGraph({
  title: "error rate",
  metricNames: [
    "error_rate",
    "failure_rate",
    "errors",
    "error_count",
    "failures",
  ],
  xLabel: "step",
  decimals: 3,
})
