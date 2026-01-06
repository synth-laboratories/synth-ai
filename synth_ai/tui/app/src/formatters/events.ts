/**
 * Event formatting and filtering helpers.
 */
import type { JobEvent } from "../tui_data"

export function formatEventData(data: unknown): string {
  if (data == null) return ""
  if (typeof data === "string") return data
  if (typeof data === "number" || typeof data === "boolean") return String(data)
  try {
    const text = JSON.stringify(data)
    return text.length > 120 ? `${text.slice(0, 117)}...` : text
  } catch {
    return String(data)
  }
}

function safeEventDataText(data: unknown): string {
  if (data == null) return ""
  if (typeof data === "string") return data
  if (typeof data === "number" || typeof data === "boolean") return String(data)
  try {
    return JSON.stringify(data)
  } catch {
    return ""
  }
}

function eventMatchesFilter(event: JobEvent, filter: string): boolean {
  const haystack = [
    event.type,
    event.message,
    event.timestamp,
    event.data ? safeEventDataText(event.data) : "",
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
  return haystack.includes(filter)
}

function eventSortKey(event: JobEvent): number {
  if (Number.isFinite(event.seq)) {
    return Number(event.seq)
  }
  const ts = event.timestamp
  if (typeof ts === "string") {
    const normalized = ts.trim().replace(" ", "T").replace(/(\\.\\d{3})\\d+/, "$1")
    const parsed = Date.parse(normalized)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return 0
}

export function getFilteredEvents(events: JobEvent[], filterText: string): JobEvent[] {
  const filter = (filterText || "").trim().toLowerCase()
  const list = filter.length ? events.filter((e) => eventMatchesFilter(e, filter)) : events
  return [...list].sort((a, b) => eventSortKey(b) - eventSortKey(a))
}

export function formatEventCardText(
  event: JobEvent,
  opts?: { isExpanded?: boolean; isLong?: boolean },
): string {
  const seq = String(event.seq).padStart(5, " ")
  const header = `${seq} ${event.type}`
  const detail = event.message ?? formatEventData(event.data)
  if (!detail) return header
  if (opts?.isExpanded) {
    const clipped = detail.length > 900 ? `${detail.slice(0, 897)}...` : detail
    return `${header}\n${clipped}`
  }
  const trimmed =
    detail.length > 120
      ? `${detail.slice(0, 117)}...${opts?.isLong ? " (enter to view)" : ""}`
      : detail
  return `${header}\n${trimmed}`
}


