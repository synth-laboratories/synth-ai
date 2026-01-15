/**
 * Registry for Metrics panel components.
 * Maps view mode → component.
 */
import type { Component } from "solid-js"
import type { MetricsPanelProps } from "../metrics/types"

export type MetricsView = "latest" | "charts"

const registry = new Map<MetricsView, Component<MetricsPanelProps>>()

/**
 * Register a metrics panel component.
 */
export function registerMetricsPanel(
  view: MetricsView,
  component: Component<MetricsPanelProps>,
): void {
  registry.set(view, component)
}

/**
 * Get the metrics panel component for a view mode.
 */
export function getMetricsPanel(view: MetricsView): Component<MetricsPanelProps> {
  const panel = registry.get(view)
  if (!panel) {
    // Try fallback to latest
    const fallback = registry.get("latest")
    if (!fallback) {
      throw new Error(`No metrics panel registered for view: ${view}`)
    }
    return fallback
  }
  return panel
}
