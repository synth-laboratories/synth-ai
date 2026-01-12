/**
 * Shared utility for normalizing frontend URL identifiers.
 */
import type { FrontendUrlId } from "../types"

/**
 * Normalize a frontend URL to a consistent identifier (hostname).
 * Used for keying API keys and other frontend-specific settings.
 */
export function normalizeFrontendId(url: string): FrontendUrlId {
  const trimmed = url.trim()
  if (!trimmed) return "unknown"
  try {
    return new URL(trimmed).host || trimmed
  } catch {
    return trimmed.replace(/^https?:\/\//, "").replace(/\/+$/, "")
  }
}
