import { createLineGraph } from "./graph-utils"

export const TokenUsageGraph = createLineGraph({
  title: "tokens processed",
  metricNames: [
    "perf.prompt_tokens",
    "perf.completion_tokens",
    "total_tokens",
    "tokens",
    "token_count",
    "tokens_processed",
    "tokens_seen",
  ],
  xLabel: "step",
  integerValues: true,
})
