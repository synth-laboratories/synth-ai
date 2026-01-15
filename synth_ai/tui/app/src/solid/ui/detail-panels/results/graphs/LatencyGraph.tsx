import { createLineGraph } from "./graph-utils"

export const LatencyGraph = createLineGraph({
  title: "latency (ms)",
  metricNames: [
    "eval.latency_ms.avg",
    "perf.decision_ms.mean",
    "perf.rollout_duration_ms.median",
    "perf.rollout_duration_ms.p90",
    "inference.latency_ms",
    "training.duration_ms",
    "latency_ms",
    "latency",
    "request_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
  ],
  xLabel: "step",
  decimals: 1,
})
