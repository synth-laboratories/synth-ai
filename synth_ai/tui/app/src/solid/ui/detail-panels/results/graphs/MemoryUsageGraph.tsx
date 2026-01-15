import { createLineGraph } from "./graph-utils"

export const MemoryUsageGraph = createLineGraph({
  title: "memory used (mb)",
  metricNames: [
    "gpu.memory.used_mb*",
    "gpu_memory_used_mb",
    "gpu_memory_used",
    "memory_used_mb",
    "memory_used",
    "gpu_memory",
  ],
  xLabel: "step",
  integerValues: true,
})
