import { createLineGraph } from "./graph-utils"

export const CandidateCountGraph = createLineGraph({
  title: "candidates per step",
  metricNames: [
    "gepa.frontier.density",
    "candidates.total",
    "candidates.count",
    "candidate_count",
    "generation_size",
    "population_size",
    "children_per_generation",
  ],
  dataField: "archive_size",
  xLabel: "step",
  integerValues: true,
})
