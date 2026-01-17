import { createEffect, createMemo, type Accessor } from "solid-js"

import { clampOffset, computeMaxOffset, ensureVisibleRange } from "../../utils/scroll"

export type ScrollState = {
  offset: Accessor<number>
  maxOffset: Accessor<number>
  viewHeight: Accessor<number>
  scrollBy: (delta: number) => boolean
  scrollTo: (next: number) => boolean
  ensureVisible: (start: number, end: number) => boolean
}

type UseScrollStateOptions = {
  offset: Accessor<number>
  setOffset: (next: number) => void
  height: Accessor<number>
  contentHeight: Accessor<number>
}

export function useScrollState(options: UseScrollStateOptions): ScrollState {
  const viewHeight = createMemo(() => Math.max(1, Math.floor(options.height())))
  const maxOffset = createMemo(() =>
    computeMaxOffset(options.contentHeight(), viewHeight()),
  )
  const clampedOffset = createMemo(() =>
    clampOffset(options.offset(), maxOffset()),
  )

  createEffect(() => {
    const next = clampedOffset()
    if (next !== options.offset()) {
      options.setOffset(next)
    }
  })

  const scrollTo = (next: number): boolean => {
    const clamped = clampOffset(next, maxOffset())
    if (clamped === options.offset()) return false
    options.setOffset(clamped)
    return true
  }

  const scrollBy = (delta: number): boolean =>
    scrollTo(options.offset() + delta)

  const ensureVisible = (start: number, end: number): boolean => {
    const next = ensureVisibleRange(options.offset(), viewHeight(), start, end)
    return scrollTo(next)
  }

  return {
    offset: clampedOffset,
    maxOffset,
    viewHeight,
    scrollBy,
    scrollTo,
    ensureVisible,
  }
}
