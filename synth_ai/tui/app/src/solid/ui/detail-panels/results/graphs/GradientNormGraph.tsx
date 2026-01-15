import { createLineGraph } from "./graph-utils"

export const GradientNormGraph = createLineGraph({
  title: "grad norm",
  metricNames: [
    "grad_norm",
    "gradient_norm",
    "optimizer.grad_norm",
    "train.grad_norm",
  ],
  xLabel: "step",
  decimals: 3,
})
