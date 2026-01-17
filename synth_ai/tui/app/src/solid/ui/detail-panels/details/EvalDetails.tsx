/**
 * Details panel for eval jobs.
 */
import { createMemo } from "solid-js"
import type { DetailsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { getPanelContentHeight, getPanelContentWidth } from "../../../../utils/panel"
import { clampLines, wrapTextLines } from "../../../../utils/text"
import type { TextPanelComponent } from "../types"
import { formatEvalDetails } from "../../../../formatters/job-details"

export function EvalDetails(props: DetailsPanelProps) {
  const detailsText = createMemo(() => {
    const job = props.data.selectedJob
    if (!job) return "No job selected."
    return formatEvalDetails(props.data, job)
  })
  const contentWidth = createMemo(() => getPanelContentWidth(props.width))
  const contentHeight = createMemo(() => getPanelContentHeight(props.height))
  const lines = createMemo(() => wrapTextLines(detailsText(), contentWidth()))
  const visibleLines = createMemo(() => clampLines(lines(), contentHeight()))

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
      <text fg={TEXT.fg}>{visibleLines().join("\n")}</text>
    </box>
  )
}

(EvalDetails as TextPanelComponent<DetailsPanelProps>).getLines = (props, contentWidth) => {
  const job = props.data.selectedJob
  if (!job) return wrapTextLines("No job selected.", contentWidth)
  return wrapTextLines(formatEvalDetails(props.data, job), contentWidth)
}
