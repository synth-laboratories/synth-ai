import { createLineGraph } from "./graph-utils"

export const RLClipFractionGraph = createLineGraph({
  title: "clip fraction",
  metricNames: [
    "clip_fraction",
    "clip_frac",
    "rl.clip_fraction",
  ],
  xLabel: "step",
  decimals: 3,
})
