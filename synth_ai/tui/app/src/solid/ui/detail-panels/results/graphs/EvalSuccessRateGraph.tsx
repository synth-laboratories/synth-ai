import { createLineGraph } from "./graph-utils"

export const EvalSuccessRateGraph = createLineGraph({
  title: "success rate",
  metricNames: [
    "eval.win_rate",
    "eval.solve_rate",
    "success_rate",
    "pass_rate",
    "completion_rate",
    "win_rate",
    "accuracy",
  ],
  xLabel: "step",
  decimals: 3,
})
