import { createMemo } from "solid-js"

import type { ResultsPanelProps } from "./types"
import { COLORS } from "../../../theme"
import { buildCombinedHint, buildActionHint } from "../../../../input/keymap"
import { clampIndex } from "../../../utils/list"
import {
  extractGraphEvolveCandidates,
  groupCandidatesByGeneration,
  summarizeBestCandidatesByGeneration,
} from "../../../../formatters/graph-evolve"
import { GraphEvolveGenerationGraph } from "./GraphEvolveGenerationGraph"

export function GraphEvolveResultsPanel(props: ResultsPanelProps) {
  const selectedGenerationIndex = () => (props.extra?.selectedGenerationIndex as number) ?? 0

  const candidates = createMemo(() => extractGraphEvolveCandidates(props.data))
  const groups = createMemo(() => groupCandidatesByGeneration(candidates()))
  const summaries = createMemo(() => summarizeBestCandidatesByGeneration(candidates()))
  const graphHeight = createMemo(() => Math.max(2, props.height - 1))
  const selectedGroup = createMemo(() => {
    const list = groups()
    if (!list.length) return null
    const index = clampIndex(selectedGenerationIndex(), list.length)
    return list[index] ?? null
  })
  const selectedBestReward = createMemo(() => {
    const group = selectedGroup()
    if (!group || !group.candidates.length) return null
    return group.candidates.reduce(
      (best, candidate) => (candidate.reward > best ? candidate.reward : best),
      group.candidates[0]!.reward,
    )
  })

  const lineWidth = createMemo(() => Math.max(1, props.width))
  const clampLine = (text: string): string => {
    const width = lineWidth()
    if (text.length <= width) return text
    return text.slice(0, width)
  }

  const formatReward = (value: number | null): string => {
    if (value == null) return "-"
    return value.toFixed(3)
  }

  const hint = createMemo(() => {
    const focusHint = props.focused
      ? buildCombinedHint("nav.up", "nav.down", "select")
      : buildCombinedHint("focus.prev", "focus.next", "focus")
    const viewHint = buildActionHint("pane.select", "view")
    const group = selectedGroup()
    const genLabel = group ? `Gen ${group.generation}` : "Gen -"
    const rewardLabel = formatReward(selectedBestReward())
    const countLabel = group ? `${group.candidates.length} candidates` : "0 candidates"
    const line = `${genLabel} | Best ${rewardLabel} | ${countLabel} | ${props.focused ? `${focusHint} | ${viewHint}` : focusHint}`
    return clampLine(line)
  })

  return (
    <box flexDirection="column" gap={0}>
      <GraphEvolveGenerationGraph
        summaries={summaries()}
        width={props.width}
        height={graphHeight()}
        focused={props.focused}
        selectedIndex={selectedGenerationIndex()}
      />
      <text fg={COLORS.textDim}>{hint()}</text>
    </box>
  )
}
