/**
 * Modal base types and utilities.
 */
import type { AppContext } from "../context"

export type ModalController = {
  readonly isVisible: boolean
  toggle: (visible: boolean) => void
  handleKey?: (key: any) => boolean // returns true if handled
}

export function centerModal(
  width: number,
  height: number,
): { left: number; top: number; width: number; height: number } {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const left = Math.max(0, Math.floor((cols - width) / 2))
  const top = Math.max(1, Math.floor((rows - height) / 2))
  return { left, top, width, height }
}

export function wrapModalText(text: string, width: number): string[] {
  const lines: string[] = []
  for (const raw of text.split("\n")) {
    if (raw.length <= width) {
      lines.push(raw)
      continue
    }
    if (raw.trim() === "") {
      lines.push("")
      continue
    }
    let start = 0
    while (start < raw.length) {
      lines.push(raw.slice(start, start + width))
      start += width
    }
  }
  return lines
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

