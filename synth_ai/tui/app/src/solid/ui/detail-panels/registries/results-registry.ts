/**
 * Registry for Results panel components.
 * Maps job type → component.
 */
import type { Component } from "solid-js"
import type { ResultsPanelProps } from "../results/types"
import type { PanelRegistryKey } from "../types"

const registry = new Map<string, Component<ResultsPanelProps>>()

/**
 * Register a results panel component.
 */
export function registerResultsPanel(
  key: string,
  component: Component<ResultsPanelProps>,
): void {
  registry.set(key, component)
}

/**
 * Get the results panel component for a job.
 * Lookup order: specific match → training type → job source → fallback
 */
export function getResultsPanel(key: PanelRegistryKey): Component<ResultsPanelProps> {
  // Try specific match: trainingType:graphType
  const specific = `${key.trainingType}:${key.graphType || "*"}`
  if (registry.has(specific)) return registry.get(specific)!

  // Try training type with wildcard
  const typeMatch = `${key.trainingType}:*`
  if (registry.has(typeMatch)) return registry.get(typeMatch)!

  // Try job source
  if (key.jobSource) {
    const sourceMatch = `${key.jobSource}:*`
    if (registry.has(sourceMatch)) return registry.get(sourceMatch)!
  }

  // Fallback
  const fallback = registry.get("*:*")
  if (!fallback) {
    throw new Error("No fallback results panel registered")
  }
  return fallback
}
