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
