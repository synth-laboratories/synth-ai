/**
 * Default details panel (fallback).
 */
import { createMemo } from "solid-js"
import type { DetailsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { getPanelContentHeight, getPanelContentWidth } from "../../../../utils/panel"
import { clampLines, wrapTextLines } from "../../../../utils/text"
import type { TextPanelComponent } from "../types"
import { formatDetails } from "../../../../formatters/job-details"

export function DefaultDetails(props: DetailsPanelProps) {
  const detailsText = createMemo(() => formatDetails(props.data))
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

(DefaultDetails as TextPanelComponent<DetailsPanelProps>).getLines = (props, contentWidth) =>
  wrapTextLines(formatDetails(props.data), contentWidth)
