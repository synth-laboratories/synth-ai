import { type Accessor, createMemo } from "solid-js"

import { computeListWindowFromStart, resolveSelectionWindow, type ListWindowItem } from "../utils/list"

export type ListWindowState<T> = {
  visibleCount: Accessor<number>
  visibleItems: Accessor<ListWindowItem<T>[]>
  windowStart: Accessor<number>
  windowEnd: Accessor<number>
  total: Accessor<number>
}

export type UseListWindowOptions<T> = {
  items: Accessor<T[]>
  selectedIndex: Accessor<number>
  height: Accessor<number>
  rowHeight: number
  chromeHeight?: number
}

export function useListWindow<T>(options: UseListWindowOptions<T>): ListWindowState<T> {
  const chromeHeight = options.chromeHeight ?? 2
  const visibleCount = createMemo(() => {
    const height = options.height()
    const rows = Math.floor((height - chromeHeight) / options.rowHeight)
    return Math.max(0, rows)
  })
  const total = createMemo(() => options.items().length)

  const windowStart = createMemo(() => {
    const items = options.items()
    const totalCount = items.length
    const visible = visibleCount()
    if (!totalCount || visible <= 0) {
      return 0
    }

    const resolved = resolveSelectionWindow(
      totalCount,
      options.selectedIndex(),
      0,
      visible,
      "edge",
    )
    return resolved.windowStart
  })

  const visibleItems = createMemo(() =>
    computeListWindowFromStart(options.items(), windowStart(), visibleCount()),
  )
  const windowEnd = createMemo(() => windowStart() + visibleItems().length)

  return {
    visibleCount,
    visibleItems,
    windowStart: windowStart,
    windowEnd,
    total,
  }
}
