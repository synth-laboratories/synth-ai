/**
 * Event list rendering + navigation helpers.
 */
import { BoxRenderable, TextRenderable } from "@opentui/core"
import type { AppContext } from "../context"
import { formatEventCardText, formatEventData, getFilteredEvents } from "../formatters"

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

export function getEventLayoutMetrics(ctx: AppContext): {
  collapsedHeight: number
  expandedHeight: number
  gap: number
  visibleCount: number
} {
  const { config } = ctx.state
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const compact = rows < 32
  const collapsedHeight = compact ? 3 : 4
  const expandedHeight = compact ? 5 : 7
  const gap = compact ? 0 : 1
  const available = Math.max(1, rows - 24)
  const maxVisible = Math.max(1, Math.floor((available + gap) / (collapsedHeight + gap)))
  const target = Math.max(1, config.eventVisibleCount)
  const visibleCount = Math.max(1, Math.min(target, maxVisible))
  return { collapsedHeight, expandedHeight, gap, visibleCount }
}

export function renderEventCards(ctx: AppContext): void {
  const { ui, renderer } = ctx
  const { snapshot, appState, config } = ctx.state

  const { collapsedHeight, expandedHeight, gap, visibleCount } = getEventLayoutMetrics(ctx)
  const recentAll = getFilteredEvents(snapshot.events, appState.eventFilter)

  if (recentAll.length === 0) {
    ui.eventsList.visible = false
    ui.eventsEmptyText.visible = true

    const job = snapshot.selectedJob
    if (appState.eventFilter) {
      ui.eventsEmptyText.content = "No events match filter."
    } else if (
      job?.status === "succeeded" ||
      job?.status === "failed" ||
      job?.status === "completed"
    ) {
      ui.eventsEmptyText.content =
        "No events recorded for this job.\n\nEvents may not have been persisted during execution."
    } else if (job?.status === "running" || job?.status === "queued") {
      ui.eventsEmptyText.content = "Waiting for events...\n\nEvents will appear as the job progresses."
    } else {
      ui.eventsEmptyText.content = "No events yet."
    }
    return
  }

  const total = recentAll.length
  const effectiveVisible = Math.max(1, visibleCount)
  appState.selectedEventIndex = clamp(appState.selectedEventIndex, 0, Math.max(0, total - 1))
  appState.eventWindowStart = clamp(
    appState.eventWindowStart,
    0,
    Math.max(0, total - effectiveVisible),
  )
  if (appState.selectedEventIndex < appState.eventWindowStart) {
    appState.eventWindowStart = appState.selectedEventIndex
  } else if (appState.selectedEventIndex >= appState.eventWindowStart + effectiveVisible) {
    appState.eventWindowStart = appState.selectedEventIndex - effectiveVisible + 1
  }

  const recent = recentAll.slice(appState.eventWindowStart, appState.eventWindowStart + effectiveVisible)

  ui.eventsEmptyText.visible = false
  ui.eventsList.visible = true
  ui.eventsList.gap = gap

  for (const card of ui.eventCards) {
    ui.eventsList.remove(card.box.id)
  }
  ui.eventCards = []

  recent.forEach((event, index) => {
    const globalIndex = appState.eventWindowStart + index
    const isSelected = globalIndex === appState.selectedEventIndex
    const detail = event.message ?? formatEventData(event.data)
    const isLong = detail.length > config.eventCollapseLimit
    const isExpanded = !!event.expanded || (isSelected && !isLong)
    const cardHeight = isExpanded ? expandedHeight : collapsedHeight

    const box = new BoxRenderable(renderer, {
      id: `event-card-${index}`,
      width: "auto",
      height: cardHeight,
      borderStyle: "single",
      borderColor: isSelected ? "#60a5fa" : "#1f2a44",
      backgroundColor: isSelected ? "#0f172a" : "#0b1220",
      border: true,
    })
    const text = new TextRenderable(renderer, {
      id: `event-card-text-${index}`,
      content: formatEventCardText(event, { isExpanded, isLong }),
      fg: "#e2e8f0",
    })
    box.add(text)
    ui.eventsList.add(box)
    ui.eventCards.push({ box, text })
  })
}

export function moveEventSelection(ctx: AppContext, delta: number): void {
  const { snapshot, appState, config } = ctx.state
  const filtered = getFilteredEvents(snapshot.events, appState.eventFilter)
  if (!filtered.length) return

  const total = filtered.length
  const visibleCount = Math.max(1, config.eventVisibleCount)
  
  // Update selected index
  const newSelected = clamp(
    appState.selectedEventIndex + delta,
    0,
    Math.max(0, total - 1),
  )
  appState.selectedEventIndex = newSelected
  
  // Update window start to keep selection visible
  let windowStart = appState.eventWindowStart
  if (newSelected < windowStart) {
    windowStart = newSelected
  } else if (newSelected >= windowStart + visibleCount) {
    windowStart = newSelected - visibleCount + 1
  }
  appState.eventWindowStart = clamp(windowStart, 0, Math.max(0, total - visibleCount))
}

export function toggleSelectedEventExpanded(ctx: AppContext): void {
  const { snapshot, appState, config } = ctx.state
  const recent = getFilteredEvents(snapshot.events, appState.eventFilter)
  const event = recent[appState.selectedEventIndex]
  if (!event) return
  const detail = event.message ?? formatEventData(event.data)
  if (detail.length <= config.eventCollapseLimit) return
  event.expanded = !event.expanded
}




