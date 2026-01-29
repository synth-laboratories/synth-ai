/**
 * Interactive Session API functions.
 *
 * Provides functions for managing OpenCode interactive sessions,
 * including local development connections and remote container sessions.
 */

import { apiGet, apiPost } from "./client"
import { isAbortError } from "../utils/abort"
import type { SessionRecord, ConnectLocalResponse, SessionHealthResult } from "../types"
import type { AppContext } from "../context"

/**
 * Fetch all interactive sessions from backend.
 */
export async function fetchSessions(
  stateFilter?: string,
  options: { signal?: AbortSignal } = {},
): Promise<SessionRecord[]> {
  try {
    const query = stateFilter ? `?state=${stateFilter}` : ""
    const sessions = await apiGet(`/interactive/sessions${query}`, options)
    return sessions || []
  } catch (err: any) {
    if (isAbortError(err)) return []
    console.error("Failed to fetch sessions:", err?.message || err)
    return []
  }
}

/**
 * Get details for a specific session.
 */
export async function getSession(
  sessionId: string,
  options: { signal?: AbortSignal } = {},
): Promise<SessionRecord | null> {
  try {
    const session = await apiGet(`/interactive/sessions/${sessionId}`, options)
    return session || null
  } catch (err: any) {
    if (isAbortError(err)) return null
    console.error(`Failed to get session ${sessionId}:`, err?.message || err)
    return null
  }
}

/**
 * Connect to a locally running OpenCode server.
 *
 * This is the lightweight local development mode where:
 * - OpenCode is running locally (user started `opencode serve`)
 * - TUI connects directly to localhost
 * - No containers or tunnels involved
 * - Backend only used for AI inference
 */
export async function connectLocal(
  opencode_url: string = "http://localhost:3000",
  model: string = "gpt-4o-mini",
  sessionId?: string,
  options: { signal?: AbortSignal } = {},
): Promise<ConnectLocalResponse> {
  const body: Record<string, any> = {
    opencode_url,
    model,
  }
  if (sessionId) {
    body.session_id = sessionId
  }
  return await apiPost("/interactive/connect-local", body, options)
}

/**
 * Disconnect from an interactive session.
 */
export async function disconnectSession(
  sessionId: string,
  options: { signal?: AbortSignal } = {},
): Promise<{ session_id: string; disconnected: boolean }> {
  return await apiPost("/interactive/disconnect", { session_id: sessionId }, options)
}

/**
 * Check health of a session's OpenCode server.
 */
export async function checkSessionHealth(
  session: SessionRecord,
  timeout: number = 5000,
  options: { signal?: AbortSignal } = {},
): Promise<SessionHealthResult> {
  const url = session.opencode_url || session.access_url
  if (!url) {
    return { healthy: false, error: "No access URL", checked_at: new Date() }
  }

  const healthUrl = `${url}/health`
  const startTime = Date.now()

  let timeoutId: ReturnType<typeof setTimeout> | null = null
  const controller = new AbortController()
  const abortExternal = () => controller.abort()

  try {
    timeoutId = setTimeout(() => controller.abort(), timeout)
    if (options.signal) {
      if (options.signal.aborted) {
        controller.abort()
      } else {
        options.signal.addEventListener("abort", abortExternal, { once: true })
      }
    }

    const response = await fetch(healthUrl, {
      signal: controller.signal,
      method: "GET",
    })

    const elapsed = Date.now() - startTime

    if (response.status === 200) {
      return { healthy: true, response_time_ms: elapsed, checked_at: new Date() }
    } else if (response.status === 404 || response.status === 405) {
      // Health endpoint not found but server responds - still healthy
      return { healthy: true, response_time_ms: elapsed, error: "Health endpoint not found", checked_at: new Date() }
    } else {
      return { healthy: false, response_time_ms: elapsed, error: `Status ${response.status}`, checked_at: new Date() }
    }
  } catch (err: any) {
    if (isAbortError(err)) {
      return { healthy: false, error: "Cancelled", checked_at: new Date() }
    }
    const elapsed = Date.now() - startTime
    const errorMessage = err?.name === "AbortError"
      ? `Timeout after ${timeout}ms`
      : err?.message || "Unknown error"

    return { healthy: false, error: errorMessage, response_time_ms: elapsed, checked_at: new Date() }
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    options.signal?.removeEventListener("abort", abortExternal)
  }
}

/**
 * Refresh sessions in app context.
 */
export async function refreshSessions(
  ctx: AppContext,
  options: { signal?: AbortSignal } = {},
): Promise<boolean> {
  const { snapshot } = ctx.state

  try {
    snapshot.sessionsLoading = true
    ctx.render()

    const sessions = await fetchSessions(undefined, options)
    snapshot.sessions = sessions
    snapshot.sessionsLoading = false
    return true
  } catch (err: any) {
    if (isAbortError(err)) return false
    snapshot.sessionsLoading = false
    return false
  }
}
