import { type Accessor, createMemo } from "solid-js"

import type { AppData } from "../../types"
import type { AppState } from "../../state/app-state"
import type { JobEvent } from "../../tui_data"
import { getFilteredEvents } from "../../formatters"
import { deriveSelectedIndex, uniqueById } from "../utils/list"
import { getEventKey } from "../../utils/events"
import { useListWindow, type ListWindowState } from "./useListWindow"

export type JobsDetailState = {
  events: Accessor<JobEvent[]>
  listWindow: ListWindowState<JobEvent>
  selectedIndex: Accessor<number>
}

type UseJobsDetailStateOptions = {
  data: AppData
  ui: AppState
  layoutHeight: Accessor<number>
}

const EVENT_PANEL_HEIGHT = 14
const EVENT_CARD_HEIGHT = 2

export function useJobsDetailState(options: UseJobsDetailStateOptions): JobsDetailState {
  const events = createMemo(() =>
    getFilteredEvents(options.data.events, options.ui.eventFilter),
  )
  const uniqueEvents = createMemo(() => uniqueById(events(), getEventKey))

  const selectedIndex = createMemo(() =>
    deriveSelectedIndex(uniqueEvents(), options.ui.selectedEventId, getEventKey),
  )

  // Fixed height to match JobsDetail panel
  const eventsHeight = createMemo(() => EVENT_PANEL_HEIGHT)

  const listWindow = useListWindow({
    items: uniqueEvents,
    selectedIndex,
    height: eventsHeight,
    rowHeight: EVENT_CARD_HEIGHT,
    chromeHeight: 2,
  })

  return {
    events: uniqueEvents,
    listWindow,
    selectedIndex,
  }
}
