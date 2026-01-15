/**
 * Metrics panel showing latest values.
 */
import { createMemo } from "solid-js"
import type { MetricsPanelProps } from "./types"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"
import { formatMetrics } from "../../../../formatters/metrics"

export function MetricsLatestPanel(props: MetricsPanelProps) {
  const metricsText = createMemo(() => formatMetrics(props.metrics))

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
