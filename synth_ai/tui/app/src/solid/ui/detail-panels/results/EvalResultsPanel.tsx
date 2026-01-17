/**
 * Results panel for eval jobs.
 */
import { createMemo } from "solid-js"
import type { ResultsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { getPanelContentHeight, getPanelContentWidth } from "../../../../utils/panel"
import { clampLines, wrapTextLines } from "../../../../utils/text"
import type { TextPanelComponent } from "../types"
import { formatEvalResults } from "../../../../formatters/results"

export function EvalResultsPanel(props: ResultsPanelProps) {
  const resultsText = createMemo(() => {
    const job = props.data.selectedJob
    if (!job) return "Results: -"
    return formatEvalResults(props.data, job)
  })
  const contentWidth = createMemo(() => getPanelContentWidth(props.width))
  const contentHeight = createMemo(() => getPanelContentHeight(props.height))
  const lines = createMemo(() => wrapTextLines(resultsText(), contentWidth()))
  const visibleLines = createMemo(() => clampLines(lines(), contentHeight()))

  return (
    <box
      border={PANEL.border}
      borderStyle={PANEL.borderStyle}
      borderColor={getPanelBorderColor(props.focused)}
      title="Results"
      titleAlignment={PANEL.titleAlignment}
      paddingLeft={PANEL.paddingLeft}
      height={props.height}
    >
      <text fg={TEXT.fg}>{visibleLines().join("\n")}</text>
    </box>
  )
}

(EvalResultsPanel as TextPanelComponent<ResultsPanelProps>).getLines = (props, contentWidth) => {
  const job = props.data.selectedJob
  if (!job) return wrapTextLines("Results: -", contentWidth)
  return wrapTextLines(formatEvalResults(props.data, job), contentWidth)
}
