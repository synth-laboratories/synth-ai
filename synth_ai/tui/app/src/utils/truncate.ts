/**
 * String truncation utilities.
 */

export function truncate(value: string, max: number): string {
  if (value.length <= max) return value
  return value.slice(0, max - 1) + "…"
}

export function truncatePath(value: string, max: number): string {
  if (value.length <= max) return value
  return "…" + value.slice(-(max - 1))
}

export function maskKey(key: string): string {
  if (!key) return "(empty)"
  if (key.length <= 8) return "sk_****"
  return key.slice(0, 6) + "…" + key.slice(-4)
}

export function maskKeyPrefix(key: string): string {
  if (!key) return "(none)"
  if (key.length <= 12) return "****"
  return key.slice(0, 8) + "…"
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

export function wrapModalText(text: string, width: number): string[] {
  const lines: string[] = []
  for (const line of text.split("\n")) {
    if (line.length <= width) {
      lines.push(line)
    } else {
      let remaining = line
      while (remaining.length > width) {
        lines.push(remaining.slice(0, width))
        remaining = remaining.slice(width)
      }
      if (remaining) lines.push(remaining)
    }
  }
  return lines
}
