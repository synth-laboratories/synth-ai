/**
 * Event navigation helpers.
 */
import type { AppContext } from "../context"
import { getFilteredEvents } from "../formatters"
import { moveSelectionById, uniqueById } from "./list"

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
  const { data, ui } = ctx.state
  const { setUi } = ctx
  const filtered = getFilteredEvents(data.events, ui.eventFilter)
  const unique = uniqueById(filtered, getEventKey)
  if (!unique.length) return

  const nextId = moveSelectionById(unique, ui.selectedEventId, delta, getEventKey)
  if (nextId && nextId !== ui.selectedEventId) {
    setUi("selectedEventId", nextId)
  }
}
