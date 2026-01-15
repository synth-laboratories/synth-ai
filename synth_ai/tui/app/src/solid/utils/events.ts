/**
 * Event navigation helpers (Solid UI).
 */
import type { AppContext } from "../../context"
import { getFilteredEvents } from "../../formatters"
import { moveSelectionById } from "./list"

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
  if (!filtered.length) return

  const nextId = moveSelectionById(filtered, ui.selectedEventId, delta, getEventKey)
  if (nextId && nextId !== ui.selectedEventId) {
    setUi("selectedEventId", nextId)
  }
}
