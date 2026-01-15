/**
 * Props for Results panel components.
 */
import type { AppData } from "../../../../types"

export interface ResultsPanelProps {
  data: AppData
  width: number
  height: number
  focused: boolean
  /** Panel-specific props (e.g., selectedGenerationIndex for graph-evolve) */
  extra?: Record<string, unknown>
}
