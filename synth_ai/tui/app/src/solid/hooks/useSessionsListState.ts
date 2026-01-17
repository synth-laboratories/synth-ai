import { createEffect, createMemo, type Accessor } from "solid-js"
import type { SetStoreFunction } from "solid-js/store"

import type { AppData, SessionRecord } from "../../types"
import type { AppState } from "../../state/app-state"
import { formatTimestamp } from "../formatters/time"
import { deriveSelectedIndex, moveSelectionById, uniqueById } from "../utils/list"
import { type ListWindowState, useListWindow } from "./useListWindow"

export type SessionsListRow = {
  id: string
  title: string
  detail: string
}

export type SessionsListState = {
  listWindow: ListWindowState<SessionsListRow>
  selectedIndex: Accessor<number>
  selectedSession: Accessor<SessionRecord | null>
  listTitle: Accessor<string>
  totalCount: Accessor<number>
  moveSelection: (delta: number) => boolean
  selectCurrent: () => boolean
}

type UseSessionsListStateOptions = {
  data: AppData
  ui: AppState
  setUi: SetStoreFunction<AppState>
  height: Accessor<number>
  isActive: Accessor<boolean>
}

function formatSessionRow(session: SessionRecord): SessionsListRow {
  const shortId = session.session_id.slice(-6)
  const location = session.is_local ? "local" : "remote"
  const state = session.state || "unknown"
  const timestamp = session.last_activity || session.connected_at || session.created_at
  const timeLabel = timestamp ? formatTimestamp(timestamp) : "-"
  return {
    id: session.session_id,
    title: `${shortId} ${location} ${state}`,
    detail: `${session.mode || "default"} · ${timeLabel}`,
  }
}

export function useSessionsListState(options: UseSessionsListStateOptions): SessionsListState {
  const sessions = createMemo(() => options.data.sessions)
  const uniqueSessions = createMemo(() =>
    uniqueById(sessions(), (session) => session.session_id),
  )
  const listItems = createMemo(() => uniqueSessions().map(formatSessionRow))
  const selectedIndex = createMemo(() =>
    deriveSelectedIndex(
      uniqueSessions(),
      options.ui.openCodeSessionId,
      (session) => session.session_id,
    ),
  )
  const listWindow = useListWindow({
    items: listItems,
    selectedIndex,
    height: options.height,
    rowHeight: 2,
    chromeHeight: 2,
  })
  const selectedSession = createMemo(() => {
    const list = uniqueSessions()
    return list[selectedIndex()] ?? null
  })
  const totalCount = createMemo(() => uniqueSessions().length)
  const listTitle = createMemo(() => {
    const total = totalCount()
    if (total <= 0) return options.data.sessionsLoading ? "Sessions (loading)" : "Sessions"
    return `Sessions [${selectedIndex() + 1}/${total}]`
  })

  const moveSelection = (delta: number): boolean => {
    const list = uniqueSessions()
    if (!list.length) return false
    const nextId = moveSelectionById(
      list,
      options.ui.openCodeSessionId,
      delta,
      (session) => session.session_id,
    )
    if (!nextId || nextId === options.ui.openCodeSessionId) return false
    options.setUi("openCodeSessionId", nextId)
    return true
  }

  const selectCurrent = (): boolean => {
    const list = uniqueSessions()
    const session = list[selectedIndex()]
    if (!session?.session_id) return false
    if (session.session_id === options.ui.openCodeSessionId) return false
    options.setUi("openCodeSessionId", session.session_id)
    return true
  }

  createEffect(() => {
    if (!options.isActive()) return
    const list = uniqueSessions()
    if (!list.length) return
    const currentId = options.ui.openCodeSessionId
    if (currentId && list.some((session) => session.session_id === currentId)) return
    options.setUi("openCodeSessionId", list[0].session_id)
  })

  return {
    listWindow,
    selectedIndex,
    selectedSession,
    listTitle,
    totalCount,
    moveSelection,
    selectCurrent,
  }
}
