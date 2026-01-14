/**
 * Event navigation helpers (Solid UI).
 */
import type { AppContext } from "../../context"
import { getFilteredEvents } from "../../formatters"
import { moveSelectionById, resolveSelectionIndexById, resolveSelectionWindow } from "./list"

export function getEventKey(event: { seq?: number; timestamp?: string | null; type?: string }): string {
  if (Number.isFinite(event.seq)) {
    return String(event.seq)
  }
  if (event.timestamp) {
    return event.timestamp
  }
  return event.type || "event"
}

export function moveEventSelection(ctx: AppContext, delta: number): void {
  const { data, ui, config } = ctx.state
  const { setUi } = ctx
  const filtered = getFilteredEvents(data.events, ui.eventFilter)
  if (!filtered.length) return

  const nextId = moveSelectionById(filtered, ui.selectedEventId, delta, getEventKey)
  if (!nextId) return
  const total = filtered.length
  const visibleCount = Math.max(1, ui.eventVisibleCount || config.eventVisibleCount)
  const nextSelected = resolveSelectionIndexById(
    filtered,
    nextId,
    getEventKey,
    ui.selectedEventIndex,
  )
  const window = resolveSelectionWindow(
    total,
    nextSelected,
    ui.eventWindowStart,
    visibleCount,
  )
  if (nextId !== ui.selectedEventId) {
    setUi("selectedEventId", nextId)
  }
  if (window.selectedIndex !== ui.selectedEventIndex) {
    setUi("selectedEventIndex", window.selectedIndex)
  }
  if (window.windowStart !== ui.eventWindowStart) {
    setUi("eventWindowStart", window.windowStart)
  }
}
