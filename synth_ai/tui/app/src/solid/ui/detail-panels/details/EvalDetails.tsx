/**
 * Details panel for eval jobs.
 */
import { createMemo } from "solid-js"
import type { DetailsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { formatEvalDetails } from "../../../../formatters/job-details"

export function EvalDetails(props: DetailsPanelProps) {
  const detailsText = createMemo(() => {
    const job = props.data.selectedJob
    if (!job) return "No job selected."
    return formatEvalDetails(props.data, job)
  })

  return (
    <box
      border={PANEL.border}
      borderStyle={PANEL.borderStyle}
      borderColor={getPanelBorderColor(props.focused ?? false)}
      title="Details"
      titleAlignment={PANEL.titleAlignment}
      paddingLeft={PANEL.paddingLeft}
      height={props.height}
    >
      <text fg={TEXT.fg}>{detailsText()}</text>
    </box>
  )
}
