/**
 * Props for Events panel components.
 */
import type { JobEvent } from "../../../../tui_data"
import type { ListWindowItem } from "../../../utils/list"
import type { JSX } from "solid-js"

export interface EventsPanelProps {
  eventItems: ListWindowItem<JobEvent>[]
  totalEvents: number
  selectedIndex: number
  focused: boolean
  width: number
  height?: number
  emptyFallback?: JSX.Element
}
