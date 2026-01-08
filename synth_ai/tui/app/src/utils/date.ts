/**
 * Date formatting utilities.
 */

/**
 * Format a date string for display, omitting year if it's the current year.
 */
export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return ""
  const d = new Date(dateStr)
  if (isNaN(d.getTime())) return ""

  const currentYear = new Date().getFullYear()
  const opts: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }
  if (d.getFullYear() !== currentYear) {
    opts.year = "numeric"
  }
  return d.toLocaleString("en-US", opts)
}

/**
 * Format a timestamp as relative time (e.g., "2 hours ago").
 */
export function formatRelativeTime(timestampMs: number): string {
  const now = Date.now()
  const diffMs = now - timestampMs
  const diffSeconds = Math.floor(diffMs / 1000)
  const diffMinutes = Math.floor(diffSeconds / 60)
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffSeconds < 60) {
    return "just now"
  } else if (diffMinutes < 60) {
    return `${diffMinutes} minute${diffMinutes === 1 ? "" : "s"} ago`
  } else if (diffHours < 24) {
    return `${diffHours} hour${diffHours === 1 ? "" : "s"} ago`
  } else if (diffDays < 7) {
    return `${diffDays} day${diffDays === 1 ? "" : "s"} ago`
  } else {
    const d = new Date(timestampMs)
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" })
  }
}
