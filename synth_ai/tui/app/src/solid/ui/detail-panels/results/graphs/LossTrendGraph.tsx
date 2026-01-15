import { createStackedLineGraph } from "./graph-utils"

export const LossTrendGraph = createStackedLineGraph([
  {
    title: "train loss",
    metricNames: [
      "train.loss",
      "training.loss",
      "sft.loss",
      "supervised.loss",
      "train_loss",
      "loss_train",
      "loss",
    ],
    xLabel: "step",
    decimals: 4,
  },
  {
    title: "val loss",
    metricNames: [
      "val.loss",
      "validation.loss",
      "eval.loss",
      "val_loss",
      "validation_loss",
      "loss_val",
    ],
    xLabel: "step",
    decimals: 4,
  },
])
