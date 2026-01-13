/**
 * Event navigation helpers (Solid UI).
 */
import type { AppContext } from "../../context"
import { getFilteredEvents } from "../../formatters"

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

export function moveEventSelection(ctx: AppContext, delta: number): void {
  const { snapshot, appState, config } = ctx.state
  const filtered = getFilteredEvents(snapshot.events, appState.eventFilter)
  if (!filtered.length) return

  const total = filtered.length
  const visibleCount = Math.max(1, config.eventVisibleCount)

  const newSelected = clamp(
    appState.selectedEventIndex + delta,
    0,
    Math.max(0, total - 1),
  )
  appState.selectedEventIndex = newSelected

  let windowStart = appState.eventWindowStart
  if (newSelected < windowStart) {
    windowStart = newSelected
  } else if (newSelected >= windowStart + visibleCount) {
    windowStart = newSelected - visibleCount + 1
  }
  appState.eventWindowStart = clamp(windowStart, 0, Math.max(0, total - visibleCount))
}
