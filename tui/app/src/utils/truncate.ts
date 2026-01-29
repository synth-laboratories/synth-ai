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

/**
 * Format an error message to fit within a modal.
 * Handles common error patterns and wraps/truncates to fit.
 */
export function formatErrorMessage(error: string, maxWidth: number = 64, maxLines: number = 3): string[] {
  let msg = error

  // Clean up common error formats
  // Auth errors - simplify the verbose dict output
  if (msg.includes("bad_auth") || msg.includes("auth_failed")) {
    msg = "Authentication failed - check SYNTH_API_KEY or run 'synth auth'"
  }

  // Word-wrap to fit width
  const lines: string[] = []
  while (msg.length > 0) {
    if (msg.length <= maxWidth) {
      lines.push(msg)
      break
    }
    // Find a good break point (space)
    let breakAt = msg.lastIndexOf(" ", maxWidth)
    if (breakAt <= 0) breakAt = maxWidth
    lines.push(msg.slice(0, breakAt))
    msg = msg.slice(breakAt).trimStart()

    // Limit lines and add ellipsis if truncated
    if (lines.length >= maxLines) {
      if (msg.length > 0) {
        const lastLine = lines[maxLines - 1]
        lines[maxLines - 1] = lastLine.slice(0, Math.max(0, maxWidth - 3)) + "..."
      }
      break
    }
  }
  return lines
}
