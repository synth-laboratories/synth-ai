/**
 * Tunnel/Task App API functions.
 */

import { apiGetV1 } from "./client"
import { isAbortError } from "../utils/abort"
import type { TunnelRecord, TunnelHealthResult } from "../types"
import type { AppContext } from "../context"

/**
 * Fetch active tunnels from backend.
 */
export async function fetchTunnels(
  statusFilter: string = "active",
  options: { signal?: AbortSignal } = {},
): Promise<TunnelRecord[]> {
  try {
    const tunnels = await apiGetV1(`/tunnels/?status_filter=${statusFilter}`, options)
    return tunnels || []
  } catch (err: any) {
    if (isAbortError(err)) return []
    console.error("Failed to fetch tunnels:", err?.message || err)
    return []
  }
}

/**
 * Refresh tunnels in app context and update snapshot.
 */
export async function refreshTunnels(
  ctx: AppContext,
  options: { signal?: AbortSignal } = {},
): Promise<boolean> {
  const { snapshot } = ctx.state

  try {
    snapshot.tunnelsLoading = true
    const tunnels = await fetchTunnels("active", options)
    snapshot.tunnels = tunnels
    snapshot.tunnelsLoading = false
    return true
  } catch (err: any) {
    if (isAbortError(err)) return false
    snapshot.tunnelsLoading = false
    return false
  }
}

/**
 * Refresh tunnel health checks for all tunnels in app context.
 * Performs client-side health checks (direct HTTP to tunnel endpoints).
 */
export async function refreshTunnelHealth(
  ctx: AppContext,
  options: { signal?: AbortSignal } = {},
): Promise<void> {
  const { snapshot } = ctx.state

  if (snapshot.tunnels.length === 0) return

  const results = await checkAllTunnelsHealth(snapshot.tunnels, 5000, 15, options)
  snapshot.tunnelHealthResults = results
}

/**
 * Check health of a single tunnel by making HTTP request to its /health endpoint.
 */
export async function checkTunnelHealth(
  tunnel: TunnelRecord,
  timeout: number = 5000,
  options: { signal?: AbortSignal } = {},
): Promise<TunnelHealthResult> {
  const url = tunnel.hostname.startsWith("http")
    ? tunnel.hostname
    : `https://${tunnel.hostname}`
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

    // Health status logic:
    // - 200: Healthy
    // - 404/405: Healthy (endpoint missing but tunnel works)
    // - 502/503: Unhealthy (backend not ready)
    // - 530: Unhealthy (Cloudflare error - tunnel not connected)
    const statusCode = response.status

    if (statusCode === 200) {
      return {
        healthy: true,
        status_code: statusCode,
        response_time_ms: elapsed,
        checked_at: new Date(),
      }
    } else if (statusCode === 404 || statusCode === 405) {
      return {
        healthy: true,
        status_code: statusCode,
        response_time_ms: elapsed,
        error: "Health endpoint not found (tunnel working)",
        checked_at: new Date(),
      }
    } else if (statusCode === 530) {
      return {
        healthy: false,
        status_code: statusCode,
        response_time_ms: elapsed,
        error: "Tunnel not connected (530)",
        checked_at: new Date(),
      }
    } else {
      return {
        healthy: false,
        status_code: statusCode,
        response_time_ms: elapsed,
        error: `Unhealthy status: ${statusCode}`,
        checked_at: new Date(),
      }
    }
  } catch (err: any) {
    if (isAbortError(err)) {
      return {
        healthy: false,
        error: "Cancelled",
        response_time_ms: 0,
        checked_at: new Date(),
      }
    }
    const elapsed = Date.now() - startTime
    const errorMessage = err?.name === "AbortError"
      ? `Timeout after ${timeout}ms`
      : err?.message || "Unknown error"

    return {
      healthy: false,
      error: errorMessage,
      response_time_ms: elapsed,
      checked_at: new Date(),
    }
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    options.signal?.removeEventListener("abort", abortExternal)
  }
}

/**
 * Check health of all tunnels in parallel.
 */
export async function checkAllTunnelsHealth(
  tunnels: TunnelRecord[],
  timeout: number = 5000,
  maxConcurrent: number = 15,
  options: { signal?: AbortSignal } = {},
): Promise<Map<string, TunnelHealthResult>> {
  const results = new Map<string, TunnelHealthResult>()

  // Process in batches to limit concurrency
  for (let i = 0; i < tunnels.length; i += maxConcurrent) {
    const batch = tunnels.slice(i, i + maxConcurrent)
    const batchResults = await Promise.all(
      batch.map(async (tunnel) => {
        if (options.signal?.aborted) {
          return { id: tunnel.id, result: { healthy: false, error: "Cancelled", checked_at: new Date() } }
        }
        const result = await checkTunnelHealth(tunnel, timeout, options)
        return { id: tunnel.id, result }
      })
    )

    for (const { id, result } of batchResults) {
      results.set(id, result)
    }
  }

  return results
}
