import { createMemo } from "solid-js"

import type { AppData } from "../../../types"
import { extractBestCandidate } from "../../../formatters/results"
import { buildPromptDiffBlocks, formatPromptDiffBlocks } from "../../../utils/prompt-diff"
import { getPanelContentHeight, getPanelContentWidth } from "../../../utils/panel"
import { clampLines, wrapTextLines } from "../../../utils/text"
import { PANEL, TEXT, getPanelBorderColor } from "../../theme"
import type { TextPanelComponent } from "./types"

export type PromptDiffPanelProps = {
  data: AppData
  width: number
  height: number
  focused?: boolean
}

type PromptDiffSelection = {
  id: string
  blocks: ReturnType<typeof buildPromptDiffBlocks>
}

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

function isPromptLearningJob(data: AppData): boolean {
  const job = data.selectedJob
  if (!job) return false
  const trainingType =
    typeof job.training_type === "string" ? job.training_type.toLowerCase() : ""
  const algo = typeof job.algorithm === "string" ? job.algorithm.toLowerCase() : ""
  const source = job.job_source ?? ""
  return (
    source === "prompt-learning" ||
    trainingType.includes("prompt") ||
    trainingType.includes("gepa") ||
    trainingType.includes("mipro") ||
    algo.includes("prompt") ||
    algo.includes("gepa") ||
    algo.includes("mipro")
  )
}

function extractCandidateId(payload: Record<string, any>, fallback: string): string {
  return (
    payload.candidate_id ||
    payload.version_id ||
    payload.template_id ||
    payload.id ||
    fallback
  )
}

function resolvePromptDiffSelection(data: AppData): PromptDiffSelection | null {
  if (data.bestSnapshot) {
    const bestCandidate = extractBestCandidate(data.bestSnapshot)
    if (bestCandidate && isRecord(bestCandidate)) {
      const blocks = buildPromptDiffBlocks(bestCandidate)
      if (blocks.length > 0) {
        const id = extractCandidateId(bestCandidate, "best")
        return { id, blocks }
      }
    }
  }

  for (let idx = data.allCandidates.length - 1; idx >= 0; idx -= 1) {
    const candidate = data.allCandidates[idx]
    const payload = isRecord(candidate.payload) ? candidate.payload : {}
    const blocks = buildPromptDiffBlocks(payload)
    if (blocks.length > 0) {
      return { id: candidate.id, blocks }
    }
  }

  return null
}

function buildPromptDiffText(data: AppData): string {
  if (!isPromptLearningJob(data)) {
    return "Prompt diff is not available for this job."
  }

  const selection = resolvePromptDiffSelection(data)
  if (!selection) {
    return "No prompt diff available yet."
  }

  const lines = [`Candidate: ${selection.id}`, "", ...formatPromptDiffBlocks(selection.blocks)]
  return lines.join("\n")
}

export function shouldShowPromptDiffPanel(data: AppData): boolean {
  return isPromptLearningJob(data)
}

export function PromptDiffPanel(props: PromptDiffPanelProps) {
  const contentWidth = createMemo(() => getPanelContentWidth(props.width))
  const contentHeight = createMemo(() => getPanelContentHeight(props.height))
  const rawText = createMemo(() => buildPromptDiffText(props.data))
  const lines = createMemo(() => wrapTextLines(rawText(), contentWidth()))
  const visibleLines = createMemo(() => clampLines(lines(), contentHeight()))

  return (
    <box
      border={PANEL.border}
      borderStyle={PANEL.borderStyle}
      borderColor={getPanelBorderColor(props.focused ?? false)}
      title="Prompt Diff"
      titleAlignment={PANEL.titleAlignment}
      paddingLeft={PANEL.paddingLeft}
      height={props.height}
    >
      <text fg={TEXT.fg}>{visibleLines().join("\n")}</text>
    </box>
  )
}

(PromptDiffPanel as TextPanelComponent<PromptDiffPanelProps>).getLines = (props, contentWidth) => {
  return wrapTextLines(buildPromptDiffText(props.data), contentWidth)
}
