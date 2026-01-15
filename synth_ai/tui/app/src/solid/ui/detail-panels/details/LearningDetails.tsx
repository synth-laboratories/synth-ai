/**
 * Details panel for learning (fine-tuning) jobs.
 */
import { createMemo } from "solid-js"
import type { DetailsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { formatLearningDetails } from "../../../../formatters/job-details"

export function LearningDetails(props: DetailsPanelProps) {
  const detailsText = createMemo(() => {
    const job = props.data.selectedJob
    if (!job) return "No job selected."
    return formatLearningDetails(job)
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
