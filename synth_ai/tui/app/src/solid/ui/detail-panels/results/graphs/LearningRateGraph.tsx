import { createLineGraph } from "./graph-utils"

export const LearningRateGraph = createLineGraph({
  title: "learning rate",
  metricNames: [
    "learning_rate",
    "lr",
    "train.lr",
    "optimizer.lr",
    "schedule.lr",
  ],
  xLabel: "step",
  decimals: 6,
})
