import { createLineGraph } from "./graph-utils"

export const FrontierSeedsSolvedGraph = createLineGraph({
  title: "frontier seeds solved",
  metricNames: [
    "gepa.frontier.total_seeds_solved",
    "frontier.total_seeds_solved",
    "frontier.seeds_solved",
    "frontier_seeds_solved",
  ],
  xLabel: "step",
  integerValues: true,
})
