import { createEffect, createMemo, createSignal, For } from "solid-js"

import type { AppData } from "../../../types"
import { num } from "../../../tui_data"
import { COLORS } from "../../theme"
import { formatActionKeys } from "../../../input/keymap"
import { clampIndex, resolveSelectionWindow } from "../../utils/list"
import {
  GOLD_TARGET,
  extractGraphEvolveCandidates,
  formatRaceLine,
  summarizeBestCandidatesByGeneration,
} from "../../../formatters/graph-evolve"

type GraphEvolveResultsPanelProps = {
  data: AppData
  width: number
  height: number
  focused: boolean
  selectedGenerationIndex: number
}

function clampWidth(value: number, min: number): number {
  return Math.max(min, value)
}

function formatBestReward(value: number | null): string {
  if (value == null) return "-"
  return value.toFixed(3)
}

export function GraphEvolveResultsPanel(props: GraphEvolveResultsPanelProps) {
  const candidates = createMemo(() => extractGraphEvolveCandidates(props.data))
  const summaries = createMemo(() => summarizeBestCandidatesByGeneration(candidates()))
  const bestRewardValue = createMemo(() => {
    const job: any = props.data.selectedJob
    const best = num(job?.best_reward ?? job?.best_score)
    if (best != null) return best
    const list = candidates()
    if (!list.length) return null
    return Math.max(...list.map((candidate) => candidate.reward))
  })
  const bestDelta = createMemo(() => {
    const best = bestRewardValue()
    return best == null ? null : Math.abs(GOLD_TARGET - best)
  })
  const scorePrecision = 2
  const listHeight = createMemo(() => Math.max(1, props.height - 4))
  const selectedIndex = createMemo(() => clampIndex(props.selectedGenerationIndex, summaries().length))
  const [windowStart, setWindowStart] = createSignal(0)

  createEffect(() => {
    const total = summaries().length
    if (total === 0) {
      setWindowStart(0)
      return
    }
    const window = resolveSelectionWindow(
      total,
      selectedIndex(),
      windowStart(),
      listHeight(),
      "edge",
    )
    if (window.windowStart !== windowStart()) {
      setWindowStart(window.windowStart)
    }
  })

  const listWindow = createMemo(() =>
    resolveSelectionWindow(
      summaries().length,
      selectedIndex(),
      windowStart(),
      listHeight(),
      "edge",
    ),
  )

  const labelWidth = createMemo(() => {
    const list = summaries()
    const maxGen = list.reduce((max, item) => Math.max(max, item.generation), 0)
    return Math.max(8, `> Gen ${maxGen}`.length)
  })

  const scoreBoxWidth = createMemo(() => {
    const list = summaries()
    const maxReward = list.reduce((max, item) => Math.max(max, item.reward), 0)
    return Math.max(6, maxReward.toFixed(scorePrecision).length + 2)
  })

  const trackWidth = createMemo(() =>
    clampWidth(props.width - labelWidth() - scoreBoxWidth() - 3, 8),
  )

  const maxDelta = createMemo(() => {
    const list = summaries()
    if (!list.length) return 0.0001
    return Math.max(...list.map((item) => item.delta), 0.0001)
  })

  const previewLines = createMemo(() => {
    const list = summaries()
    if (!list.length) {
      return ["No candidates yet."]
    }
    const window = listWindow()
    const lines: string[] = []
    for (let i = window.windowStart; i < window.windowEnd; i += 1) {
      const item = list[i]
      if (!item) continue
      const cursor = i === selectedIndex() ? ">" : " "
      const label = `${cursor} Gen ${item.generation}`
      lines.push(formatRaceLine({
        label,
        reward: item.reward,
        delta: item.delta,
        maxDelta: maxDelta(),
        trackWidth: trackWidth(),
        labelWidth: labelWidth(),
        scorePrecision,
      }))
    }
    return lines
  })

  const header = createMemo(() => {
    const bestReward = formatBestReward(bestRewardValue())
    const delta = bestDelta()
    const deltaLabel = delta == null ? "-" : delta.toFixed(2)
    return `Gold ${GOLD_TARGET.toFixed(2)} | Best ${bestReward} (d=${deltaLabel})`
  })

  const listTitle = createMemo(() => {
    const window = listWindow()
    if (!window.total) return "Best per gen"
    const range =
      window.total > window.visibleCount
        ? ` [${window.windowStart + 1}-${window.windowEnd}/${window.total}]`
        : ` (${window.total})`
    return `Best per gen${range}`
  })

  const hint = createMemo(() => {
    const focusHint = props.focused
      ? `${formatActionKeys("nav.up", { primaryOnly: true })}/${formatActionKeys("nav.down", { primaryOnly: true })} scroll`
      : `${formatActionKeys("focus.prev", { primaryOnly: true })}/${formatActionKeys("focus.next", { primaryOnly: true })} focus`
    const viewHint = `${formatActionKeys("pane.select", { primaryOnly: true })} view`
    return `${focusHint} | ${viewHint}`
  })

  return (
    <box flexDirection="column" gap={0}>
      <text fg={COLORS.textDim}>{`Status: ${props.data.selectedJob?.status ?? "-"}`}</text>
      <text fg={COLORS.text}>{header()}</text>
      <text fg={COLORS.textDim}>{listTitle()}</text>
      <For each={previewLines()}>
        {(line) => <text fg={COLORS.text}>{line}</text>}
      </For>
      <text fg={COLORS.textDim}>{hint()}</text>
    </box>
  )
}
