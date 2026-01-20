/**
 * Time/date formatting utilities.
 */

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

/**
 * Format date as "Jan 8 at 10:32 AM" (gold reference style)
 */
function formatDateGoldStyle(date: Date): string {
  const month = MONTHS[date.getMonth()]
  const day = date.getDate()
  let hours = date.getHours()
  const minutes = date.getMinutes().toString().padStart(2, "0")
  const ampm = hours >= 12 ? "PM" : "AM"
  hours = hours % 12
  if (hours === 0) hours = 12
  return `${month} ${day} at ${hours}:${minutes} ${ampm}`
}

/**
 * Parse any timestamp value to a Date object
 */
function parseToDate(value: any): Date | null {
  if (value == null || value === "") return null
  if (value instanceof Date) return value

  if (typeof value === "object") {
    const seconds = (value as any).seconds
    const nanos = (value as any).nanoseconds ?? (value as any).nanos
    if (Number.isFinite(Number(seconds))) {
      const ms = Number(seconds) * 1000 + (Number(nanos) || 0) / 1e6
      return new Date(ms)
    }
  }

  if (typeof value === "number") {
    const ms = value > 1e12 ? value : value * 1000
    return new Date(ms)
  }

  if (typeof value === "string") {
    const trimmed = value.trim()
    const normalized = trimmed
      .replace(" ", "T")
      .replace(/(\.\d{3})\d+/, "$1")
    const parsed = Date.parse(normalized)
    if (Number.isFinite(parsed)) {
      return new Date(parsed)
    }
    if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) {
      const numeric = Number(trimmed)
      const ms = numeric > 1e12 ? numeric : numeric * 1000
      return new Date(ms)
    }
    const numericMatch = trimmed.match(/-?\d+(?:\.\d+)?/)
    if (numericMatch) {
      const parsedNumber = Number(numericMatch[0])
      if (Number.isFinite(parsedNumber)) {
        const ms = parsedNumber > 1e12 ? parsedNumber : parsedNumber * 1000
        return new Date(ms)
      }
    }
  }

  return null
}

export function formatTimestamp(value: any): string {
  const date = parseToDate(value)
  if (!date) return "-"
  return formatDateGoldStyle(date)
}

/** Full timestamp for details panels */
export function formatTimestampFull(value: any): string {
  const date = parseToDate(value)
  if (!date) return "-"
  return date.toLocaleString()
}

export function formatValue(value: unknown): string {
  // Keep behavior aligned with the legacy monolith to avoid UI diffs.
  if (value == null) return "-"
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(4) : String(value)
  }
  if (typeof value === "string") return value
  if (typeof value === "boolean") return value ? "true" : "false"
  try {
    const text = JSON.stringify(value)
    return text.length > 120 ? `${text.slice(0, 117)}...` : text
  } catch {
    return String(value)
  }
}
