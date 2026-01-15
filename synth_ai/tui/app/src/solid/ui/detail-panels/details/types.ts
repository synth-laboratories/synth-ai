/**
 * Props for Details panel components.
 */
import type { AppData } from "../../../../types"

export interface DetailsPanelProps {
  data: AppData
  width: number
  height: number
  focused?: boolean
}
