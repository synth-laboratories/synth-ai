import { createLineGraph } from "./graph-utils"

export const ThroughputGraph = createLineGraph({
  title: "throughput (tok/s)",
  metricNames: [
    "perf.tokens_sec",
    "perf.tokens_sec_engine",
    "perf.tokens_sec.req_avg",
    "perf.tokens_sec.per_gpu",
    "tokens_per_second",
    "token_per_second",
    "throughput.tokens_per_second",
    "samples_per_second",
    "steps_per_second",
    "tokens_per_sec",
  ],
  xLabel: "step",
  integerValues: true,
})
