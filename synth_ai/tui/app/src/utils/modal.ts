import { wrapTextLines } from "./text"

export type ScrollableModalView = {
  lines: string[]
  visible: string[]
  offset: number
  maxOffset: number
  bodyHeight: number
}

export function buildScrollableModal(
  raw: string,
  width: number,
  height: number,
  offset: number,
): ScrollableModalView {
  // Account for borders (2) + padding left/right (4) = 6 chars of horizontal chrome
  const maxWidth = Math.max(10, width - 6)
  const lines = wrapTextLines(raw, maxWidth)
  // Account for: 2 (borders) + 2 (padding top/bottom) + 1 (title) + 1 (hint) = 6 lines of chrome
  const bodyHeight = Math.max(1, height - 6)
  const maxOffset = Math.max(0, lines.length - bodyHeight)
  const clamped = clamp(offset, 0, maxOffset)
  const visible = lines.slice(clamped, clamped + bodyHeight)
  return { lines, visible, offset: clamped, maxOffset, bodyHeight }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}
