/**
 * Task App Discovery API
 *
 * Combines tunnels from backend with locally scanned task apps.
 */

import { spawn } from "child_process"
import { appState } from "../state/app-state"
import type { TunnelRecord, TunnelHealthResult, ScannedApp, TaskApp } from "../types"

// Cache scan results to avoid excessive CLI calls
let scanCache: { apps: ScannedApp[]; timestamp: number } | null = null
const CACHE_TTL_MS = 30_000 // 30 seconds

/**
 * Scan for local task apps using the synth-ai scan CLI command.
 * Results are cached for 30 seconds.
 */
export async function scanLocalApps(): Promise<ScannedApp[]> {
  // Return cached results if still valid
  if (scanCache && Date.now() - scanCache.timestamp < CACHE_TTL_MS) {
    return scanCache.apps
  }

  const cwd = appState.opencodeWorkingDir || process.cwd()
  const portRange = process.env.SYNTH_SCAN_PORT_RANGE || "8000-8100"

  return new Promise((resolve) => {
    try {
      const proc = spawn("synth-ai", ["scan", "--json", "--port-range", portRange], {
        cwd,
        stdio: ["ignore", "pipe", "pipe"],
      })

      let stdout = ""
      let stderr = ""

      proc.stdout?.on("data", (data: Buffer) => {
        stdout += data.toString()
      })

      proc.stderr?.on("data", (data: Buffer) => {
        stderr += data.toString()
      })

      proc.on("error", () => {
        // synth-ai scan not available, return empty
        resolve([])
      })

      proc.on("close", (code) => {
        if (code !== 0) {
          // Scan failed, return empty
          resolve([])
          return
        }

        try {
          // Parse JSON output - may be multiple JSON objects or array
          const apps: ScannedApp[] = JSON.parse(stdout)
          scanCache = { apps, timestamp: Date.now() }
          resolve(apps)
        } catch {
          // Parse failed, return empty
          resolve([])
        }
      })

      // Timeout after 10 seconds
      setTimeout(() => {
        proc.kill("SIGTERM")
        resolve([])
      }, 10_000)
    } catch {
      resolve([])
    }
  })
}

/**
 * Clear the scan cache to force a fresh scan.
 */
export function clearScanCache(): void {
  scanCache = null
}

/**
 * Merge tunnels and scanned apps into a unified TaskApp list.
 * Deduplicates by URL, preferring tunnels over scanned apps.
 */
export function mergeTaskApps(
  tunnels: TunnelRecord[],
  tunnelHealth: Map<string, TunnelHealthResult>,
  scannedApps: ScannedApp[]
): TaskApp[] {
  const apps: TaskApp[] = []
  const seenUrls = new Set<string>()

  // Add tunnels first (they have priority)
  for (const tunnel of tunnels) {
    const url = `https://${tunnel.hostname}`
    seenUrls.add(url)

    const health = tunnelHealth.get(tunnel.id)
    const healthStatus = health?.healthy
      ? "healthy"
      : health
        ? "unhealthy"
        : "unknown"

    apps.push({
      id: `tunnel-${tunnel.id}`,
      name: tunnel.hostname.split(".")[0],
      url,
      type: "tunnel",
      health_status: healthStatus,
    })
  }

  // Add scanned local apps (skip duplicates)
  for (const app of scannedApps) {
    if (seenUrls.has(app.url)) continue
    seenUrls.add(app.url)

    apps.push({
      id: `local-${app.port || app.name}`,
      name: app.name || app.task_name || `port-${app.port}`,
      url: app.url,
      type: "local",
      port: app.port,
      health_status: app.health_status,
      discovered_via: app.discovered_via,
    })
  }

  return apps
}

/**
 * Check health of a single task app.
 */
export async function checkTaskAppHealth(
  app: TaskApp,
  timeout: number = 5000
): Promise<{ healthy: boolean; response_time_ms?: number; error?: string }> {
  const healthUrl = `${app.url}/health`
  const startTime = Date.now()

  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    const response = await fetch(healthUrl, {
      signal: controller.signal,
      method: "GET",
    })

    clearTimeout(timeoutId)
    const elapsed = Date.now() - startTime

    // Health status logic (same as tunnels)
    const statusCode = response.status
    if (statusCode === 200 || statusCode === 404 || statusCode === 405) {
      return { healthy: true, response_time_ms: elapsed }
    } else {
      return { healthy: false, response_time_ms: elapsed, error: `Status ${statusCode}` }
    }
  } catch (err: any) {
    const elapsed = Date.now() - startTime
    const errorMessage = err?.name === "AbortError"
      ? `Timeout after ${timeout}ms`
      : err?.message || "Unknown error"
    return { healthy: false, response_time_ms: elapsed, error: errorMessage }
  }
}
