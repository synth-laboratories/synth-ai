import { createLineGraph } from "./graph-utils"

export const GpuUtilizationGraph = createLineGraph({
  title: "gpu utilization",
  metricNames: [
    "gpu.utilization.percent*",
    "gpu_utilization",
    "gpu_util",
    "gpu_usage",
    "utilization_gpu",
  ],
  xLabel: "step",
  decimals: 1,
})
