/**
 * Results panel for eval jobs.
 */
import { createMemo } from "solid-js"
import type { ResultsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { formatEvalResults } from "../../../../formatters/results"

export function EvalResultsPanel(props: ResultsPanelProps) {
  const resultsText = createMemo(() => {
    const job = props.data.selectedJob
    if (!job) return "Results: -"
    return formatEvalResults(props.data, job)
  })

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
      <text fg={TEXT.fg}>{resultsText()}</text>
    </box>
  )
}
