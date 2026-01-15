/**
 * Metrics panel showing charts.
 */
import { createMemo } from "solid-js"
import type { MetricsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { formatMetricsCharts } from "../../../../formatters/metrics"

export function MetricsChartsPanel(props: MetricsPanelProps) {
  const metricsText = createMemo(() =>
    formatMetricsCharts(props.metrics, {
      width: Math.max(30, props.width - 6),
      height: props.height,
    }),
  )

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
      <text fg={TEXT.fg}>{metricsText()}</text>
    </box>
  )
}
