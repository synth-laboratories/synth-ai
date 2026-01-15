import { createMemo } from "solid-js"

import type { AppData } from "../../../../../types"
import { renderLineChart, type MetricPoint } from "../../../../../formatters/ascii_chart"
import { COLORS } from "../../../../theme"

export type TrainingGraphProps = {
  data: AppData
  width: number
  height: number
  focused?: boolean
}

type MetricSeriesSpec = {
  title: string
  metricNames: string[]
  xLabel?: string
  decimals?: number
  integerValues?: boolean
  pointChar?: string
  dataField?: string
}

type HistogramSpec = {
  title: string
  metricNames: string[]
  bins?: number
  decimals?: number
  dataField?: string
}

const DEFAULT_POINT = "*"

type SeriesPoint = { x: number; y: number; meta?: unknown }

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function toNumber(value: unknown): number | null {
  if (isFiniteNumber(value)) return value
  if (typeof value === "string") {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function metricPoints(data: AppData): MetricPoint[] {
  const metrics: any = data.metrics || {}
  const points = Array.isArray(metrics?.points) ? metrics.points : []
  return points as MetricPoint[]
}

function resolveMetricName(points: MetricPoint[], name: string): string | null {
  if (!name.endsWith("*")) return name
  const prefix = name.slice(0, -1)
  if (!prefix) return null
  const matches = new Set<string>()
  for (const p of points) {
    if (!p || typeof p !== "object") continue
    const raw = String((p as any).name ?? "")
    if (raw.startsWith(prefix)) matches.add(raw)
  }
  if (matches.size === 0) return null
  return Array.from(matches).sort((a, b) => a.length - b.length || a.localeCompare(b))[0]
}

function extractSeriesMatching(
  points: MetricPoint[],
  name: string,
  dataField?: string,
): SeriesPoint[] {
  const resolved = resolveMetricName(points, name)
  if (!resolved) return []
  const out: SeriesPoint[] = []
  let fallbackX = 0
  for (const p of points) {
    if (!p || typeof p !== "object") continue
    if (String((p as any).name ?? "") !== resolved) continue
    const data: any = (p as any).data
    const raw = dataField ? data?.[dataField] : (p as any).value
    const y = toNumber(raw)
    if (y == null) continue
    const step = toNumber((p as any).step)
    const x = step != null ? step : fallbackX
    fallbackX += 1
    out.push({ x, y, meta: data })
  }
  out.sort((a, b) => a.x - b.x)
  return out
}

function pickSeries(points: MetricPoint[], spec: MetricSeriesSpec) {
  for (const name of spec.metricNames) {
    let series = extractSeriesMatching(points, name, spec.dataField)
    if (series.length === 0 && spec.dataField) {
      series = extractSeriesMatching(points, name)
    }
    if (series.length > 0) return series
  }
  return []
}

function renderSeriesChart(
  points: MetricPoint[],
  spec: MetricSeriesSpec,
  width: number,
  plotHeight: number,
): string {
  const series = pickSeries(points, spec)
  return renderLineChart(series, {
    width,
    height: plotHeight,
    title: spec.title,
    xLabel: spec.xLabel,
    decimals: spec.decimals,
    integerValues: spec.integerValues,
    pointChar: spec.pointChar ?? DEFAULT_POINT,
  }).text
}

function clampTextLines(text: string, height: number): string {
  const maxLines = Math.max(1, height)
  const lines = text.split("\n")
  if (lines.length >= maxLines) return lines.slice(0, maxLines).join("\n")
  return lines.join("\n")
}

function plotHeightForStack(totalHeight: number, count: number, gap: number): number {
  if (count <= 0) return 2
  const reserved = count * 3 + Math.max(0, count - 1) * gap
  const available = Math.max(0, totalHeight - reserved)
  const perChart = Math.max(1, Math.floor(available / count))
  return Math.max(2, perChart)
}

function pickStackCount(totalHeight: number, count: number, gap: number): number {
  const minChartLines = 5
  if (count <= 1) return count
  const maxCharts = Math.floor((totalHeight + gap) / (minChartLines + gap))
  return Math.max(1, Math.min(count, maxCharts))
}

function stackCharts(charts: string[], height: number, gap: number): string {
  if (charts.length === 0) return ""
  const joiner = "\n".repeat(Math.max(1, gap + 1))
  return clampTextLines(charts.join(joiner), height)
}

function collectValues(points: MetricPoint[], spec: HistogramSpec): number[] {
  for (const name of spec.metricNames) {
    let series = extractSeriesMatching(points, name, spec.dataField)
    if (series.length === 0 && spec.dataField) {
      series = extractSeriesMatching(points, name)
    }
    if (series.length > 0) {
      return series.map((point) => point.y).filter((value) => Number.isFinite(value))
    }
  }
  return []
}

function renderHistogram(options: {
  title: string
  values: number[]
  width: number
  height: number
  bins?: number
  decimals?: number
}): string {
  const width = Math.max(18, Math.floor(options.width))
  const height = Math.max(3, Math.floor(options.height))
  const title = options.title.trim() || "distribution"
  const values = options.values
  if (values.length === 0) {
    return `${title}\n(no data yet)`
  }

  const decimals = Math.max(0, options.decimals ?? 2)
  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  const span = maxValue - minValue

  const maxBins = Math.max(1, Math.min(12, height - 2))
  const binCount = Math.max(1, Math.min(options.bins ?? maxBins, maxBins))
  const counts = Array.from({ length: binCount }, () => 0)

  if (span === 0) {
    counts[binCount - 1] = values.length
  } else {
    for (const value of values) {
      const t = (value - minValue) / span
      const idx = Math.min(binCount - 1, Math.floor(t * binCount))
      counts[idx] += 1
    }
  }

  const fmt = (value: number) => value.toFixed(decimals)
  const ranges = counts.map((_, idx) => {
    const start = minValue + (span * idx) / binCount
    const end = minValue + (span * (idx + 1)) / binCount
    return { start, end }
  })
  const labels = ranges.map((range) => `${fmt(range.start)}-${fmt(range.end)}`)
  const labelWidth = Math.max(...labels.map((label) => label.length))
  const barWidth = Math.max(1, width - labelWidth - 3)
  const maxCount = Math.max(...counts)

  const clampLine = (line: string) => line.slice(0, width)
  const header = `${title}  n=${values.length}  min=${fmt(minValue)}  max=${fmt(maxValue)}`
  const lines = [clampLine(header)]

  for (let i = 0; i < counts.length; i += 1) {
    const count = counts[i]
    const barSize = maxCount > 0 ? Math.round((count / maxCount) * barWidth) : 0
    const bar = "#".repeat(barSize).padEnd(barWidth, " ")
    const label = labels[i]?.padStart(labelWidth, " ") ?? "".padStart(labelWidth, " ")
    lines.push(clampLine(`${label} | ${bar}`))
  }

  return clampTextLines(lines.join("\n"), height)
}

export function createLineGraph(spec: MetricSeriesSpec) {
  return function LineGraph(props: TrainingGraphProps) {
    const chartText = createMemo(() => {
      const points = metricPoints(props.data)
      const width = Math.max(18, props.width)
      const plotHeight = Math.max(2, props.height - 3)
      const chart = renderSeriesChart(points, spec, width, plotHeight)
      return clampTextLines(chart, props.height)
    })
    return <text fg={COLORS.text}>{chartText()}</text>
  }
}

export function createStackedLineGraph(series: MetricSeriesSpec[], options?: { gap?: number }) {
  return function StackedLineGraph(props: TrainingGraphProps) {
    const chartText = createMemo(() => {
      const gap = Math.max(0, options?.gap ?? 1)
      const width = Math.max(18, props.width)
      const points = metricPoints(props.data)
      const chartCount = pickStackCount(props.height, series.length, gap)
      const visibleSeries = series.slice(0, chartCount)
      const plotHeight = plotHeightForStack(props.height, visibleSeries.length, gap)
      const charts = visibleSeries.map((spec) =>
        renderSeriesChart(points, spec, width, plotHeight),
      )
      return stackCharts(charts, props.height, gap)
    })
    return <text fg={COLORS.text}>{chartText()}</text>
  }
}

export function createHistogramGraph(spec: HistogramSpec) {
  return function HistogramGraph(props: TrainingGraphProps) {
    const chartText = createMemo(() => {
      const points = metricPoints(props.data)
      const values = collectValues(points, spec).filter((value) => toNumber(value) != null)
      const width = Math.max(18, props.width)
      const chart = renderHistogram({
        title: spec.title,
        values,
        width,
        height: Math.max(3, props.height),
        bins: spec.bins,
        decimals: spec.decimals,
      })
      return clampTextLines(chart, props.height)
    })
    return <text fg={COLORS.text}>{chartText()}</text>
  }
}
