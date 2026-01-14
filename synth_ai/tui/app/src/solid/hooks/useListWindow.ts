import { type Accessor, createMemo } from "solid-js"

import { computeListWindow, type ListWindowItem } from "../utils/list"

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

  const visibleItems = createMemo(() =>
    computeListWindow(options.items(), options.selectedIndex(), visibleCount()),
  )

  const total = createMemo(() => options.items().length)
  const windowStart = createMemo(() => {
    const items = visibleItems()
    return items.length ? items[0].globalIndex : 0
  })
  const windowEnd = createMemo(() => {
    const items = visibleItems()
    return items.length ? items[items.length - 1].globalIndex + 1 : 0
  })

  return {
    visibleCount,
    visibleItems,
    windowStart,
    windowEnd,
    total,
  }
}
