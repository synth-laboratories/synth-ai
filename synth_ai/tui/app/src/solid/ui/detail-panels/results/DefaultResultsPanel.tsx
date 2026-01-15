/**
 * Default results panel for prompt-learning and other job types.
 */
import { createMemo } from "solid-js"
import type { ResultsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { formatResults } from "../../../../formatters/results"

export function DefaultResultsPanel(props: ResultsPanelProps) {
  const resultsText = createMemo(() => formatResults(props.data))

  return (
    <box
      border={PANEL.border}
      borderStyle={PANEL.borderStyle}
      borderColor={getPanelBorderColor(props.focused)}
      title="Summary"
      titleAlignment={PANEL.titleAlignment}
      paddingLeft={PANEL.paddingLeft}
      height={props.height}
    >
      <text fg={TEXT.fg}>{resultsText()}</text>
    </box>
  )
}
