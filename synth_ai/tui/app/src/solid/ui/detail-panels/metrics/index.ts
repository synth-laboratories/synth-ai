/**
 * Metrics panel exports.
 */
import { registerMetricsPanel } from "../registries/metrics-registry"
import { MetricsLatestPanel } from "./MetricsLatestPanel"
import { MetricsChartsPanel } from "./MetricsChartsPanel"

// Register panels
registerMetricsPanel("latest", MetricsLatestPanel)
registerMetricsPanel("charts", MetricsChartsPanel)

export { MetricsLatestPanel } from "./MetricsLatestPanel"
export { MetricsChartsPanel } from "./MetricsChartsPanel"
export type { MetricsPanelProps } from "./types"
