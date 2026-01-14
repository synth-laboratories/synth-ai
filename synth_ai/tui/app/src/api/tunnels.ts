/**
 * Tunnel/Task App API functions.
 */

import { apiGet } from "./client"
import { isAbortError } from "../utils/abort"
import { fetchWithTimeout, isAborted } from "../utils/request"
import type { TunnelRecord, TunnelHealthResult } from "../types"
import type { AppContext } from "../context"

/**
 * Fetch active tunnels from backend.
 */
export async function fetchTunnels(
  statusFilter: string = "active",
): Promise<TunnelRecord[]> {
  try {
    const tunnels = await apiGet(`/tunnels/?status_filter=${statusFilter}`, { version: "v1" })
    return tunnels || []
  } catch (err: any) {
    if (isAbortError(err)) return []
    console.error("Failed to fetch tunnels:", err?.message || err)
    return []
  }
}

/**
 * Refresh tunnels in app context and update data store.
 */
export async function refreshTunnels(
  ctx: AppContext,
): Promise<boolean> {
  const { setData } = ctx

  try {
    setData("tunnelsLoading", true)
    const tunnels = await fetchTunnels("active")
    setData("tunnels", tunnels)
    setData("tunnelsLoading", false)
    return true
  } catch (err: any) {
    if (isAbortError(err)) return false
    setData("tunnelsLoading", false)
    return false
  }
}

/**
 * Refresh tunnel health checks for all tunnels in app context.
 * Performs client-side health checks (direct HTTP to tunnel endpoints).
 */
export async function refreshTunnelHealth(
  ctx: AppContext,
): Promise<void> {
  const { data } = ctx.state
  const { setData } = ctx

  if (data.tunnels.length === 0) return

  const results = await checkAllTunnelsHealth(data.tunnels, 5000, 15)
  setData("tunnelHealthResults", new Map(results))
}

/**
 * Check health of a single tunnel by making HTTP request to its /health endpoint.
 */
export async function checkTunnelHealth(
  tunnel: TunnelRecord,
  timeout: number = 5000,
): Promise<TunnelHealthResult> {
  const url = tunnel.hostname.startsWith("http")
    ? tunnel.hostname
    : `https://${tunnel.hostname}`
  const healthUrl = `${url}/health`
  const startTime = Date.now()

  try {
    const response = await fetchWithTimeout(healthUrl, {
      method: "GET",
      timeoutMs: timeout,
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
    if (err?.name === "TimeoutError") {
      return {
        healthy: false,
        error: `Timeout after ${timeout}ms`,
        response_time_ms: timeout,
        checked_at: new Date(),
      }
    }
    const elapsed = Date.now() - startTime
    const errorMessage = err?.message || "Unknown error"

    return {
      healthy: false,
      error: errorMessage,
      response_time_ms: elapsed,
      checked_at: new Date(),
    }
  }
}

/**
 * Check health of all tunnels in parallel.
 */
export async function checkAllTunnelsHealth(
  tunnels: TunnelRecord[],
  timeout: number = 5000,
  maxConcurrent: number = 15,
): Promise<Map<string, TunnelHealthResult>> {
  const results = new Map<string, TunnelHealthResult>()

  // Process in batches to limit concurrency
  for (let i = 0; i < tunnels.length; i += maxConcurrent) {
    const batch = tunnels.slice(i, i + maxConcurrent)
    const batchResults = await Promise.all(
      batch.map(async (tunnel) => {
        if (isAborted()) {
          return { id: tunnel.id, result: { healthy: false, error: "Cancelled", checked_at: new Date() } }
        }
        const result = await checkTunnelHealth(tunnel, timeout)
        return { id: tunnel.id, result }
      })
    )

    for (const { id, result } of batchResults) {
      results.set(id, result)
    }
  }

  return results
}
