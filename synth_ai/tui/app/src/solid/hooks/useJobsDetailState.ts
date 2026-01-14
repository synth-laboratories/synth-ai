import { type Accessor, createEffect, createMemo } from "solid-js"

import type { AppData } from "../../types"
import type { AppState } from "../../state/app-state"
import type { JobEvent } from "../../tui_data"
import { getFilteredEvents } from "../../formatters"
import { resolveSelectionIndexById, resolveSelectionWindow } from "../utils/list"
import { getEventKey } from "../utils/events"
import type { SetStoreFunction } from "solid-js/store"

export type JobsEventWindow = {
  total: number
  visible: number
  selected: number
  windowStart: number
  slice: JobEvent[]
}

export type JobsDetailState = {
  events: Accessor<JobEvent[]>
  eventWindow: Accessor<JobsEventWindow>
}

type UseJobsDetailStateOptions = {
  data: AppData
  ui: AppState
  setUi: SetStoreFunction<AppState>
  layoutHeight: Accessor<number>
}

const EVENT_RESERVED_ROWS = 16
const EVENT_CARD_HEIGHT = 2

export function useJobsDetailState(options: UseJobsDetailStateOptions): JobsDetailState {
  const events = createMemo(() =>
    getFilteredEvents(options.data.events, options.ui.eventFilter),
  )
  const selectedIndex = createMemo(() =>
    resolveSelectionIndexById(
      events(),
      options.ui.selectedEventId,
      getEventKey,
      options.ui.selectedEventIndex,
    ),
  )
  const eventWindow = createMemo(() => {
    const list = events()
    const total = list.length
    const available = Math.max(1, options.layoutHeight() - EVENT_RESERVED_ROWS)
    const visibleRows = Math.max(1, Math.floor(available / EVENT_CARD_HEIGHT))
    const visible = visibleRows
    const window = resolveSelectionWindow(
      total,
      selectedIndex(),
      options.ui.eventWindowStart,
      visible,
    )
    return {
      total,
      visible: window.visibleCount,
      selected: window.selectedIndex,
      windowStart: window.windowStart,
      slice: list.slice(window.windowStart, window.windowEnd),
    }
  })

  createEffect(() => {
    const list = events()
    if (!list.length) {
      if (options.ui.selectedEventId !== null) {
        options.setUi("selectedEventId", null)
      }
      if (options.ui.selectedEventIndex !== 0) {
        options.setUi("selectedEventIndex", 0)
      }
      if (options.ui.eventWindowStart !== 0) {
        options.setUi("eventWindowStart", 0)
      }
      return
    }
    const selectedId = options.ui.selectedEventId
    if (!selectedId) return
    if (!list.some((event) => getEventKey(event) === selectedId)) {
      options.setUi("selectedEventId", null)
      if (options.ui.selectedEventIndex !== 0) {
        options.setUi("selectedEventIndex", 0)
      }
      if (options.ui.eventWindowStart !== 0) {
        options.setUi("eventWindowStart", 0)
      }
      return
    }
    const nextSelected = selectedIndex()
    if (nextSelected !== options.ui.selectedEventIndex) {
      options.setUi("selectedEventIndex", nextSelected)
    }
    const available = Math.max(1, options.layoutHeight() - EVENT_RESERVED_ROWS)
    const visibleRows = Math.max(1, Math.floor(available / EVENT_CARD_HEIGHT))
    const visible = visibleRows
    if (visible !== options.ui.eventVisibleCount) {
      options.setUi("eventVisibleCount", visible)
    }
    const window = resolveSelectionWindow(
      list.length,
      nextSelected,
      options.ui.eventWindowStart,
      visible,
    )
    if (window.windowStart !== options.ui.eventWindowStart) {
      options.setUi("eventWindowStart", window.windowStart)
    }
  })

  return {
    events,
    eventWindow,
  }
}
