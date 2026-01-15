/**
 * Props for Metrics panel components.
 */

export interface MetricsPanelProps {
  metrics: Record<string, unknown>
  width: number
  height: number
  focused: boolean
}
