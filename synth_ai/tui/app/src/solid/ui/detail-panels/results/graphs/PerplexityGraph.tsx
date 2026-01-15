import { createLineGraph } from "./graph-utils"

export const PerplexityGraph = createLineGraph({
  title: "perplexity",
  metricNames: [
    "perplexity",
    "train.perplexity",
    "val.perplexity",
    "validation.perplexity",
  ],
  xLabel: "step",
  decimals: 3,
})
