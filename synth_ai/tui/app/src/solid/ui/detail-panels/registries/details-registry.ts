/**
 * Registry for Details panel components.
 * Maps job type → component.
 */
import type { Component } from "solid-js"
import type { DetailsPanelProps } from "../details/types"
import type { PanelRegistryKey } from "../types"

const registry = new Map<string, Component<DetailsPanelProps>>()

/**
 * Register a details panel component.
 */
export function registerDetailsPanel(
  key: string,
  component: Component<DetailsPanelProps>,
): void {
  registry.set(key, component)
}

/**
 * Get the details panel component for a job.
 * Lookup order: specific match → training type → job source → fallback
 */
export function getDetailsPanel(key: PanelRegistryKey): Component<DetailsPanelProps> {
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
    throw new Error("No fallback details panel registered")
  }
  return fallback
}
