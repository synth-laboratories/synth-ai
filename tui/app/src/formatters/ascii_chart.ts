/**
 * Lightweight ASCII/Unicode line chart renderer for terminal UIs.
 *
 * Design goals:
 * - No deps
 * - Deterministic output
 * - Graceful handling of empty/sparse/constant series
 * - Fits within a fixed (width x height) rectangle
 */
export type MetricPoint = {
  name?: unknown
  value?: unknown
  step?: unknown
  created_at?: unknown
  data?: unknown
}

type SeriesPoint = { x: number; y: number; meta?: unknown }

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function toNumber(value: unknown): number | null {
  if (isFiniteNumber(value)) return value
  if (typeof value === "string") {
    const n = Number(value)
    return Number.isFinite(n) ? n : null
  }
  return null
}

function formatCompactNumber(value: number, decimals: number): string {
  if (!Number.isFinite(value)) return "-"
  // Avoid scientific notation for small/medium numbers.
  const fixed = value.toFixed(decimals)
  // Trim trailing zeros (but keep at least one digit after dot if decimals > 0).
  if (decimals > 0) {
    return fixed.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "")
  }
  return fixed
}

export type RenderLineChartOptions = {
  width: number
  height: number
  title: string
  xLabel?: string
  pointChar?: string
  decimals?: number
  /**
   * When true, treat values as integer-like (render without decimals).
   */
  integerValues?: boolean
}

export type RenderedChart = {
  text: string
  usedWidth: number
  usedHeight: number
}

/**
 * Extract a (step -> value) series for a given metric name from a list of points.
 * - Prefers numeric `step` when present.
 * - Falls back to monotonically increasing index ordering.
 */
export function extractSeries(points: MetricPoint[], metricName: string): SeriesPoint[] {
  const out: SeriesPoint[] = []
  let fallbackX = 0
  for (const p of points) {
    if (!p || typeof p !== "object") continue
    if (String((p as any).name || "") !== metricName) continue
    const y = toNumber((p as any).value)
    if (y == null) continue
    const step = toNumber((p as any).step)
    const x = step != null ? step : fallbackX
    fallbackX += 1
    out.push({ x, y, meta: (p as any).data })
  }
  // Sort by x ascending, stable for repeated x by insertion order.
  out.sort((a, b) => a.x - b.x)
  return out
}

/**
 * Extract a derived series from metric point `data` payload.
 * This is useful when the primary metric point carries auxiliary fields (e.g. archive_size).
 */
export function extractSeriesFromDataField(
  points: MetricPoint[],
  metricName: string,
  dataField: string,
): SeriesPoint[] {
  const out: SeriesPoint[] = []
  let fallbackX = 0
  for (const p of points) {
    if (!p || typeof p !== "object") continue
    if (String((p as any).name || "") !== metricName) continue
    const data: any = (p as any).data
    const y = toNumber(data?.[dataField])
    if (y == null) continue
    const step = toNumber((p as any).step)
    const x = step != null ? step : fallbackX
    fallbackX += 1
    out.push({ x, y, meta: data })
  }
  out.sort((a, b) => a.x - b.x)
  return out
}

type BinnedSeries = {
  bins: Array<number | null>
  present: boolean[]
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  latest: number | null
}

function binSeries(points: SeriesPoint[], width: number): BinnedSeries | null {
  if (points.length === 0) return null
  const w = Math.max(1, Math.floor(width))
  const xMin = points[0].x
  const xMax = points[points.length - 1].x
  const latest = points[points.length - 1]?.y ?? null

  const bins: Array<number | null> = Array.from({ length: w }, () => null)
  const present: boolean[] = Array.from({ length: w }, () => false)
  if (xMax === xMin) {
    // Single x — put latest in last column for visibility.
    bins[w - 1] = latest
    present[w - 1] = true
  } else {
    const span = xMax - xMin
    // We pick "last value in bin" as the representative (works well for progress).
    for (const pt of points) {
      const t = (pt.x - xMin) / span
      const col = clamp(Math.floor(t * (w - 1) + 0.000001), 0, w - 1)
      bins[col] = pt.y
      present[col] = true
    }
    // Fill gaps with carry-forward for scaling context, but mark them as non-present so we can
    // render them with a lighter glyph instead of "painting" the whole chart.
    let last: number | null = null
    for (let i = 0; i < bins.length; i += 1) {
      if (bins[i] == null) {
        bins[i] = last
      } else {
        last = bins[i]
      }
    }
  }

  const ys = bins.filter((v): v is number => v != null && Number.isFinite(v))
  if (ys.length === 0) return null
  const yMin = Math.min(...ys)
  const yMax = Math.max(...ys)
  return { bins, present, xMin, xMax, yMin, yMax, latest }
}

/**
 * Render a single series as an ASCII chart.
 *
 * Output lines:
 * - title line
 * - H plot lines (each includes y-label gutter + axis + plot)
 * - x-axis line
 * - x-label line
 */
export function renderLineChart(points: SeriesPoint[], opts: RenderLineChartOptions): RenderedChart {
  const title = (opts.title || "").trim() || "metric"
  const width = Math.max(18, Math.floor(opts.width))
  const height = Math.max(2, Math.floor(opts.height))
  const pointChar = (opts.pointChar || "•").slice(0, 1)
  const decimals = opts.integerValues ? 0 : Math.max(0, opts.decimals ?? 3)

  // Reserve a left gutter for y labels: " 0.800 " (~7-10 chars).
  const yGutter = 9
  const plotWidth = Math.max(5, width - yGutter - 2) // 2 = space + axis

  const binned = binSeries(points, plotWidth)
  if (!binned) {
    const msg = `${title}\n(no data yet)`
    return { text: msg, usedWidth: width, usedHeight: 2 }
  }

  const { bins, present, xMin, xMax, yMin, yMax, latest } = binned
  const span = yMax - yMin

  const fmt = (v: number) => formatCompactNumber(v, decimals)
  const yTop = fmt(yMax)
  const yBot = fmt(yMin)

  const grid: string[][] = Array.from({ length: height }, () =>
    Array.from({ length: plotWidth }, () => " "),
  )

  // Plot points
  for (let col = 0; col < bins.length; col += 1) {
    const y = bins[col]
    if (y == null || !Number.isFinite(y)) continue
    const row =
      span === 0
        ? Math.floor((height - 1) / 2)
        : clamp(Math.round(((yMax - y) / span) * (height - 1)), 0, height - 1)
    // Use a lighter glyph for carry-forward points so we don't draw an "infinite" dotted line
    // when the series is constant or sparse.
    grid[row][col] = present[col] ? pointChar : "·"
  }

  const lines: string[] = []
  const latestStr = latest == null ? "-" : fmt(latest)
  const xLabel = opts.xLabel ? `  ${opts.xLabel}` : ""
  lines.push(`${title}  latest=${latestStr}${xLabel}`)

  // Build plot lines with y labels on top and bottom.
  for (let r = 0; r < height; r += 1) {
    let yLabel = ""
    if (r === 0) yLabel = yTop
    else if (r === height - 1) yLabel = yBot
    const padded = yLabel.padStart(yGutter - 2, " ")
    const axis = "|"
    lines.push(`${padded} ${axis}${grid[r].join("")}`)
  }

  // X axis with tick marks
  const axisLine = Array.from({ length: plotWidth }, () => "-")
  axisLine[0] = "+" // Left corner
  axisLine[plotWidth - 1] = "+" // Right corner

  // If the x-range is small and integer-like, draw a tick at *every* step increment.
  // This is the most readable for ranges like 0..4.
  const spanX = xMax - xMin
  const xMinInt = Number.isInteger(xMin)
  const xMaxInt = Number.isInteger(xMax)
  const maxPerStepTicks = 30
  if (xMinInt && xMaxInt && spanX > 0 && spanX <= maxPerStepTicks) {
    for (let step = xMin; step <= xMax; step += 1) {
      const t = (step - xMin) / spanX
      const col = clamp(Math.round(t * (plotWidth - 1)), 0, plotWidth - 1)
      axisLine[col] = "+" // per-step tick
    }
  } else {
    // Otherwise, fall back to a few evenly spaced ticks to avoid a dense axis.
    const numTicks = Math.min(5, Math.max(3, Math.floor(plotWidth / 12)))
    const tickSpacing = plotWidth / (numTicks + 1)
    for (let i = 1; i <= numTicks; i += 1) {
      const tickPos = Math.floor(i * tickSpacing)
      if (tickPos > 0 && tickPos < plotWidth - 1) {
        axisLine[tickPos] = "+"
      }
    }
  }

  lines.push(`${" ".repeat(yGutter - 1)}${axisLine.join("")}`)
  
  // Place xMin at left, xMax at right. If they're the same, show once in the middle.
  const leftLabel = String(xMin)
  const rightLabel = String(xMax)
  const xChars = Array.from({ length: plotWidth }, () => " ")
  
  if (xMin === xMax) {
    // Single point: center the label
    const label = leftLabel
    const start = Math.max(0, Math.floor((plotWidth - label.length) / 2))
    for (let i = 0; i < label.length && start + i < plotWidth; i += 1) {
      xChars[start + i] = label[i]
    }
  } else {
    // Multiple points: xMin on left, xMax on right
    // Place left label (xMin) starting at position 0
    for (let i = 0; i < leftLabel.length && i < plotWidth; i += 1) {
      xChars[i] = leftLabel[i]
    }
    
    // Place right label (xMax) at the end, but avoid overwriting left label if too close
    const rightStart = Math.max(leftLabel.length + 1, plotWidth - rightLabel.length)
    for (let i = 0; i < rightLabel.length && rightStart + i < plotWidth; i += 1) {
      xChars[rightStart + i] = rightLabel[i]
    }
  }
  
  lines.push(`${" ".repeat(yGutter)} ${xChars.join("")}`)

  return { text: lines.join("\n"), usedWidth: width, usedHeight: lines.length }
}


