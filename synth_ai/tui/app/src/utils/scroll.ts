export function computeMaxOffset(total: number, viewHeight: number): number {
  return Math.max(0, Math.floor(total) - Math.max(1, Math.floor(viewHeight)))
}

export function clampOffset(offset: number, maxOffset: number): number {
  return Math.max(0, Math.min(Math.floor(offset), Math.floor(maxOffset)))
}

export function ensureVisibleRange(
  offset: number,
  viewHeight: number,
  start: number,
  end: number,
): number {
  const top = Math.min(start, end)
  const bottom = Math.max(start, end)
  const height = Math.max(1, Math.floor(viewHeight))
  if (top < offset) return top
  if (bottom > offset + height) return Math.max(0, bottom - height)
  return offset
}
