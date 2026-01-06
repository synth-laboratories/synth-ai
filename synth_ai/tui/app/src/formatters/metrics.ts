/**
 * Metrics formatting utilities.
 */
import { formatValue } from "./time"

export function formatMetrics(metricsValue: Record<string, any> | unknown): string {
  const metrics: any = metricsValue || {}
  const points = Array.isArray(metrics?.points) ? metrics.points : []
  if (points.length > 0) {
    const latestByName = new Map<string, any>()
    for (const point of points) {
      if (point?.name) {
        latestByName.set(String(point.name), point)
      }
    }
    const rows = Array.from(latestByName.values()).sort((a, b) =>
      String(a.name).localeCompare(String(b.name)),
    )
    if (rows.length === 0) return "Metrics: -"
    const limit = 12
    const lines = rows.slice(0, limit).map((point) => {
      const value = formatValue(point.value ?? point.data ?? "-")
      const step = point.step != null ? ` (step ${point.step})` : ""
      return `- ${point.name}: ${value}${step}`
    })
    if (rows.length > limit) {
      lines.push(`... +${rows.length - limit} more`)
    }
    return ["Metrics (latest):", ...lines].join("\n")
  }

  const keys = Object.keys(metrics).filter((k) => k !== "points" && k !== "job_id")
  if (keys.length === 0) return "Metrics: -"
  return ["Metrics:", ...keys.map((k) => `- ${k}: ${formatValue(metrics[k])}`)].join("\n")
}


