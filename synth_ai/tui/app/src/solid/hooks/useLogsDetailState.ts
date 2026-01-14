import { type Accessor, createEffect, createMemo } from "solid-js"
import type { SetStoreFunction } from "solid-js/store"

import type { AppState } from "../../state/app-state"
import type { LogFileInfo } from "../utils/logs"

export type LogsDetailView = {
  lines: string[]
  visibleLines: string[]
  offset: number
  maxOffset: number
}

export type LogsDetailState = {
  view: Accessor<LogsDetailView>
  scrollBy: (delta: number) => boolean
  jumpToTail: () => void
  onBlur: () => void
  onFocus: () => void
}

type UseLogsDetailStateOptions = {
  selectedFile: Accessor<LogFileInfo | null>
  lines: Accessor<string[]>
  height: Accessor<number>
  ui: AppState
  setUi: SetStoreFunction<AppState>
}

const LOGS_DETAIL_HEADER_ROWS = 6

export function useLogsDetailState(options: UseLogsDetailStateOptions): LogsDetailState {
  const visibleHeight = createMemo(() => Math.max(1, options.height() - LOGS_DETAIL_HEADER_ROWS))
  const maxOffset = createMemo(() =>
    Math.max(0, options.lines().length - visibleHeight()),
  )

  // Track line count when focus leaves, to detect new content on focus return
  let lineCountOnBlur = 0
  // When tail mode is on and content changes, keep offset synced to max
  createEffect(() => {
    if (options.ui.logsDetailTail) {
      const max = maxOffset()
      if (options.ui.logsDetailOffset !== max) {
        options.setUi("logsDetailOffset", max)
      }
    }
  })

  const view = createMemo((): LogsDetailView => {
    const lines = options.lines()
    if (!lines.length) {
      return { lines: [], visibleLines: [], offset: 0, maxOffset: 0 }
    }
    const max = maxOffset()
    // Always use offset as single source of truth (tail mode syncs it via effect above)
    const offset = Math.min(options.ui.logsDetailOffset, max)
    const visibleLines = lines.slice(offset, offset + visibleHeight())
    return { lines, visibleLines, offset, maxOffset: max }
  })

  const selectedPath = createMemo(() => options.selectedFile()?.path ?? null)
  createEffect((prevPath: string | null) => {
    const path = selectedPath()
    if (path && path !== prevPath) {
      options.setUi("logsDetailOffset", 0)
      options.setUi("logsDetailTail", true)
    }
    return path
  }, null as string | null)

  const scrollBy = (delta: number): boolean => {
    const lines = options.lines()
    if (!lines.length) return false
    const max = maxOffset()
    const current = options.ui.logsDetailOffset
    const next = Math.max(0, Math.min(max, current + delta))
    if (next === current) return false
    options.setUi("logsDetailOffset", next)
    options.setUi("logsDetailTail", false)
    return true
  }

  const jumpToTail = () => {
    const max = maxOffset()
    options.setUi("logsDetailOffset", max)
    options.setUi("logsDetailTail", true)
  }

  const onBlur = () => {
    lineCountOnBlur = options.lines().length
  }

  const onFocus = () => {
    const currentCount = options.lines().length
    // If new lines arrived while away, jump to tail to show them
    if (currentCount > lineCountOnBlur) {
      jumpToTail()
    }
  }

  return {
    view,
    scrollBy,
    jumpToTail,
    onBlur,
    onFocus,
  }
}
