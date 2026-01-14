import { COLORS } from "../theme"

export type ListWindowItem<T> = {
  item: T
  globalIndex: number
}

export type SelectionStyle = {
  fg: string
  bg: string | undefined
}

export type SelectionWindowMode = "edge" | "center"

export type SelectionWindow = {
  total: number
  visibleCount: number
  selectedIndex: number
  windowStart: number
  windowEnd: number
}

export function getSelectionStyle(isSelected: boolean): SelectionStyle {
  return {
    fg: isSelected ? COLORS.textSelected : COLORS.text,
    bg: isSelected ? COLORS.bgSelection : undefined,
  }
}

export function clampIndex(value: number, length: number): number {
  if (length <= 0) return 0
  return Math.min(Math.max(value, 0), length - 1)
}

export function moveSelectionIndex(current: number, delta: number, length: number): number {
  return clampIndex(current + delta, length)
}

export function resolveSelectionWindow(
  total: number,
  selectedIndex: number,
  windowStart: number,
  visibleCount: number,
  mode: SelectionWindowMode = "edge",
): SelectionWindow {
  const safeTotal = Math.max(0, total)
  const visible = Math.max(1, visibleCount)
  const selected = clampIndex(selectedIndex, safeTotal)
  const maxStart = Math.max(0, safeTotal - Math.min(safeTotal, visible))
  let start = windowStart

  if (mode === "center") {
    start = selected - Math.floor(visible / 2)
  } else {
    start = Math.max(0, Math.min(start, maxStart))
    if (selected < start) {
      start = selected
    } else if (selected >= start + visible) {
      start = selected - visible + 1
    }
  }

  start = Math.max(0, Math.min(start, maxStart))
  const end = Math.min(safeTotal, start + visible)

  return {
    total: safeTotal,
    visibleCount: visible,
    selectedIndex: selected,
    windowStart: start,
    windowEnd: end,
  }
}

export function wrapIndex(value: number, length: number): number {
  if (length <= 0) return 0
  return ((value % length) + length) % length
}

export function computeListWindow<T>(
  items: T[],
  selectedIndex: number,
  visibleCount: number,
): ListWindowItem<T>[] {
  if (!items.length || visibleCount <= 0) return []
  const maxVisible = Math.min(items.length, visibleCount)
  let start = 0
  if (selectedIndex >= start + maxVisible) {
    start = selectedIndex - maxVisible + 1
  }
  if (selectedIndex < start) {
    start = selectedIndex
  }
  return items.slice(start, start + maxVisible).map((item, idx) => ({
    item,
    globalIndex: start + idx,
  }))
}

export function deriveSelectedIndex<T>(
  items: T[],
  selectedId: string | null,
  getId: (item: T) => string,
): number {
  if (!items.length) return 0
  if (!selectedId) return 0
  const idx = items.findIndex((item) => getId(item) === selectedId)
  return idx >= 0 ? idx : 0
}

export function resolveSelectionIndexById<T>(
  items: T[],
  selectedId: string | null,
  getId: (item: T) => string,
  fallbackIndex: number,
): number {
  if (!items.length) return 0
  if (!selectedId) {
    return clampIndex(fallbackIndex, items.length)
  }
  const idx = items.findIndex((item) => getId(item) === selectedId)
  if (idx >= 0) return idx
  return clampIndex(fallbackIndex, items.length)
}

export function moveSelectionById<T>(
  items: T[],
  selectedId: string | null,
  delta: number,
  getId: (item: T) => string,
): string | null {
  if (!items.length) return null
  const currentIndex = deriveSelectedIndex(items, selectedId, getId)
  const nextIndex = wrapIndex(currentIndex + delta, items.length)
  return getId(items[nextIndex])
}
