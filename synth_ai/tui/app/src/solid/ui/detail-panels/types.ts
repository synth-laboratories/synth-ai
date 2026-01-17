/**
 * Shared types for detail panel registries.
 */

import type { Component } from "solid-js"

export type PanelRegistryKey = {
  trainingType: string | null
  graphType: string | null
  jobSource: string | null
}

export type TextPanelComponent<P extends Record<string, any>> = Component<P> & {
  getLines?: (props: P, contentWidth: number) => string[]
}
