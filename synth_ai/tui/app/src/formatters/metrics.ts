/**
 * Metrics formatting utilities.
 */
import { formatValue } from "./time"
import { extractSeries, renderLineChart, type MetricPoint } from "./ascii_chart"

function safePoints(metricsValue: Record<string, any> | unknown): MetricPoint[] {
  const metrics: any = metricsValue || {}
  const points = Array.isArray(metrics?.points) ? metrics.points : []
  return points as MetricPoint[]
}

export function formatMetricsCharts(
  metricsValue: Record<string, any> | unknown,
  opts: { width: number; height: number; isGepa?: boolean },
): string {
  const points = safePoints(metricsValue)
  if (!opts.isGepa) {
    return "Charts: (only available for GEPA jobs)"
  }
  if (!points.length) {
    // Show helpful message about fetching metrics
    const metrics: any = metricsValue || {}
    const hasPayload = metrics && typeof metrics === "object"
    const hasJobId = hasPayload && "job_id" in metrics
    const hasEmptyPoints = hasPayload && Array.isArray(metrics.points) && metrics.points.length === 0
    
    if (hasEmptyPoints) {
      return "Charts: (metrics endpoint returned empty array - backend may not have emitted metrics for this job)"
    }
    if (hasJobId && !hasPayload.points) {
      return "Charts: (no metric points yet - press 'm' to fetch metrics)"
    }
    return "Charts: (no metric points yet)"
  }
  
  // Check if we have GEPA metrics specifically
  const gepaPoints = points.filter((pt: any) => pt?.name?.startsWith("gepa."))
  if (gepaPoints.length === 0) {
    const allNames = [...new Set(points.map((pt: any) => pt?.name).filter(Boolean))].slice(0, 3)
    return `Charts: (found ${points.length} metric points, but none are GEPA metrics. Available: ${allNames.join(", ") || "none"})`
  }

  // Allocate vertical space for two stacked charts.
  // Each chart uses: 1 title + H plot + 2 axes/labels = H+3 lines.
  // With height 18, we can use H=5-6 per chart comfortably.
  const chartPlotHeight = Math.max(5, Math.floor((opts.height - 6) / 2))
  const innerWidth = Math.max(30, opts.width - 4)

  const densitySeries = extractSeries(points, "gepa.frontier.density")
  const seedsSeries = extractSeries(points, "gepa.frontier.total_seeds_solved")

  const densityChart = renderLineChart(densitySeries, {
    width: innerWidth,
    height: chartPlotHeight,
    title: "frontier density  (gepa.frontier.density)",
    xLabel: "step",
    decimals: 3,
  }).text

  const seedsChart = renderLineChart(seedsSeries, {
    width: innerWidth,
    height: chartPlotHeight,
    title: "seeds solved  (gepa.frontier.total_seeds_solved)",
    xLabel: "step",
    integerValues: true,
  }).text

  // If a chart has no data, it will render "(no data yet)".
  const parts = [densityChart, "", seedsChart]
  // Hard-trim to available height if needed.
  const lines = parts.join("\n").split("\n")
  const maxLines = Math.max(4, opts.height)
  return lines.slice(0, maxLines).join("\n")
}

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


