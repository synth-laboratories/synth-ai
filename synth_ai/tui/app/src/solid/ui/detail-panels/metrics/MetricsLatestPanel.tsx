/**
 * Metrics panel showing latest values.
 */
import { createMemo } from "solid-js"
import type { MetricsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { getPanelContentHeight, getPanelContentWidth } from "../../../../utils/panel"
import { clampLines, wrapTextLines } from "../../../../utils/text"
import type { TextPanelComponent } from "../types"
import { formatMetrics } from "../../../../formatters/metrics"

export function MetricsLatestPanel(props: MetricsPanelProps) {
  const metricsText = createMemo(() => formatMetrics(props.metrics))
  const contentWidth = createMemo(() => getPanelContentWidth(props.width))
  const contentHeight = createMemo(() => getPanelContentHeight(props.height))
  const lines = createMemo(() => wrapTextLines(metricsText(), contentWidth()))
  const visibleLines = createMemo(() => clampLines(lines(), contentHeight()))

  return (
    <box
      border={PANEL.border}
      borderStyle={PANEL.borderStyle}
      borderColor={getPanelBorderColor(props.focused)}
      title="Metrics"
      titleAlignment={PANEL.titleAlignment}
      paddingLeft={PANEL.paddingLeft}
      height={props.height}
    >
      <text fg={TEXT.fg}>{visibleLines().join("\n")}</text>
    </box>
  )
}

(MetricsLatestPanel as TextPanelComponent<MetricsPanelProps>).getLines = (props, contentWidth) =>
  wrapTextLines(formatMetrics(props.metrics), contentWidth)
