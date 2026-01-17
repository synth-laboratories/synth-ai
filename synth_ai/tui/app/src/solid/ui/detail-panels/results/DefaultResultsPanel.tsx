/**
 * Default results panel for prompt-learning and other job types.
 */
import { createMemo } from "solid-js"
import type { ResultsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { getPanelContentHeight, getPanelContentWidth } from "../../../../utils/panel"
import { clampLines, wrapTextLines } from "../../../../utils/text"
import type { TextPanelComponent } from "../types"
import { formatResults } from "../../../../formatters/results"

export function DefaultResultsPanel(props: ResultsPanelProps) {
  const resultsText = createMemo(() => formatResults(props.data))
  const contentWidth = createMemo(() => getPanelContentWidth(props.width))
  const contentHeight = createMemo(() => getPanelContentHeight(props.height))
  const lines = createMemo(() => wrapTextLines(resultsText(), contentWidth()))
  const visibleLines = createMemo(() => clampLines(lines(), contentHeight()))

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
      <text fg={TEXT.fg}>{visibleLines().join("\n")}</text>
    </box>
  )
}

(DefaultResultsPanel as TextPanelComponent<ResultsPanelProps>).getLines = (props, contentWidth) =>
  wrapTextLines(formatResults(props.data), contentWidth)
