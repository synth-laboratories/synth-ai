import { createEffect, createMemo, createSignal } from "solid-js"

import { REWARD_MAX, type GenerationSummary } from "../../../formatters/graph-evolve"
import { COLORS } from "../../theme"
import { clampIndex, resolveSelectionWindow } from "../../utils/list"

type GraphEvolveGenerationGraphProps = {
  summaries: GenerationSummary[]
  width: number
  height: number
  focused: boolean
  selectedIndex: number
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function drawLine(
  grid: string[][],
  from: { x: number; y: number },
  to: { x: number; y: number },
  char: string,
): void {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const steps = Math.max(Math.abs(dx), Math.abs(dy))
  if (steps === 0) return
  for (let i = 0; i <= steps; i += 1) {
    const x = Math.round(from.x + (dx * i) / steps)
    const y = Math.round(from.y + (dy * i) / steps)
    if (!grid[y] || grid[y]?.[x] == null) continue
    if (grid[y][x] === " ") {
      grid[y][x] = char
    }
  }
}

export function GraphEvolveGenerationGraph(props: GraphEvolveGenerationGraphProps) {
  const width = createMemo(() => Math.max(1, props.width))
  const height = createMemo(() => Math.max(1, props.height))
  const showXAxisLabel = createMemo(() => height() >= 5)
  const chartHeight = createMemo(() =>
    Math.max(1, height() - (showXAxisLabel() ? 1 : 0)),
  )
  const labelWidth = createMemo(() => Math.min(4, Math.max(3, width() - 4)))
  const axisGap = 1
  const plotWidth = createMemo(() =>
    Math.max(1, width() - labelWidth() - axisGap - 1),
  )

  const selectedIndex = createMemo(() => clampIndex(props.selectedIndex, props.summaries.length))
  const selectedSummary = createMemo(() => props.summaries[selectedIndex()] ?? null)
  const orderedSummaries = createMemo(() => {
    return [...props.summaries].sort((a, b) => a.generation - b.generation)
  })
  const selectedOrderedIndex = createMemo(() => {
    const selected = selectedSummary()
    if (!selected) return -1
    return orderedSummaries().findIndex((item) => item.generation === selected.generation)
  })

  const [windowStart, setWindowStart] = createSignal(0)

  createEffect(() => {
    const total = orderedSummaries().length
    if (total === 0) {
      setWindowStart(0)
      return
    }
    const visibleCount = Math.max(1, Math.min(plotWidth(), total))
    const selected = Math.max(0, selectedOrderedIndex())
    const window = resolveSelectionWindow(
      total,
      selected,
      windowStart(),
      visibleCount,
      "center",
    )
    if (window.windowStart !== windowStart()) {
      setWindowStart(window.windowStart)
    }
  })

  const chartLines = createMemo(() => {
    const total = orderedSummaries().length
    const w = width()
    const h = height()
    if (!total) {
      const message = "No candidates yet."
      const lines = [message.slice(0, w).padEnd(w, " ")]
      while (lines.length < h) {
        lines.push(" ".repeat(w))
      }
      return lines
    }

    const plotW = plotWidth()
    const plotH = chartHeight()
    const visibleCount = Math.max(1, Math.min(plotW, total))
    const window = resolveSelectionWindow(
      total,
      Math.max(0, selectedOrderedIndex()),
      windowStart(),
      visibleCount,
      "center",
    )
    const slice = orderedSummaries().slice(window.windowStart, window.windowEnd)
    const points = slice.map((item, idx) => {
      const maxValue = REWARD_MAX > 0 ? REWARD_MAX : 1
      const normalized = maxValue > 0 ? item.reward / maxValue : 0
      const clamped = clamp(normalized, 0, 1)
      const x = visibleCount === 1
        ? Math.floor((plotW - 1) / 2)
        : Math.round((idx / (visibleCount - 1)) * (plotW - 1))
      const y = Math.round((1 - clamped) * (plotH - 1))
      return {
        generation: item.generation,
        reward: item.reward,
        x,
        y,
      }
    })

    const grid = Array.from({ length: plotH }, () => Array(plotW).fill(" "))
    const axisRow = plotH - 1
    for (let x = 0; x < plotW; x += 1) {
      grid[axisRow][x] = "-"
    }

    for (let i = 0; i < points.length - 1; i += 1) {
      drawLine(grid, points[i], points[i + 1], ".")
    }

    const selectedGen = selectedSummary()?.generation ?? null
    for (const point of points) {
      const isSelected = selectedGen != null && point.generation === selectedGen
      const marker = isSelected && props.focused ? "@" : "o"
      grid[point.y][point.x] = marker
    }

    const lines: string[] = []
    const showMid = plotH >= 5
    const midRow = Math.round((plotH - 1) / 2)
    for (let row = 0; row < plotH; row += 1) {
      let label = ""
      if (row === 0) {
        label = "1.0"
      } else if (row === axisRow) {
        label = "0.0"
      } else if (showMid && row === midRow) {
        label = "0.5"
      }
      const paddedLabel = label.padStart(labelWidth(), " ")
      const axisChar = row === axisRow ? "+" : "|"
      const line = `${paddedLabel}${" ".repeat(axisGap)}${axisChar}${grid[row].join("")}`
      lines.push(line.slice(0, w).padEnd(w, " "))
    }

    if (showXAxisLabel()) {
      const firstGen = slice[0]?.generation
      const lastGen = slice[slice.length - 1]?.generation
      const label = firstGen != null && lastGen != null ? `Gen ${firstGen}-${lastGen}` : "Gen"
      const prefix = " ".repeat(labelWidth() + axisGap + 1)
      const line = `${prefix}${label}`
      lines.push(line.slice(0, w).padEnd(w, " "))
    }

    while (lines.length < h) {
      lines.push(" ".repeat(w))
    }
    return lines.slice(0, h)
  })

  return (
    <text fg={COLORS.text}>{chartLines().join("\n")}</text>
  )
}
