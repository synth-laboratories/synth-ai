/**
 * Default details panel (fallback).
 */
import { createMemo } from "solid-js"
import type { DetailsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { formatDetails } from "../../../../formatters/job-details"

export function DefaultDetails(props: DetailsPanelProps) {
  const detailsText = createMemo(() => formatDetails(props.data))

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
