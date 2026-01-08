/**
 * OpenCode server management - auto-start and lifecycle management.
 */
import { spawn, type ChildProcess } from "child_process"
import { registerCleanup } from "../lifecycle"
import { appState } from "../state/app-state"

let openCodeProcess: ChildProcess | null = null
let serverUrl: string | null = null

/**
 * Start the OpenCode server in the background.
 * Returns the server URL once it's ready.
 */
export async function startOpenCodeServer(): Promise<string | null> {
  // Don't start if already running
  if (openCodeProcess && !openCodeProcess.killed) {
    return serverUrl
  }

  // Check if user has opencode installed
  return new Promise((resolve) => {
    try {
      openCodeProcess = spawn("opencode", ["serve"], {
        stdio: ["ignore", "pipe", "pipe"],
        detached: false,
      })

      let resolved = false

      // Parse stdout for the server URL
      openCodeProcess.stdout?.on("data", (data: Buffer) => {
        const output = data.toString()
        // Look for "listening on http://..." pattern
        const match = output.match(/listening on (https?:\/\/[^\s]+)/)
        if (match && !resolved) {
          serverUrl = match[1]
          appState.openCodeUrl = serverUrl
          resolved = true
          resolve(serverUrl)
        }
      })

      // Also check stderr (some tools output there)
      openCodeProcess.stderr?.on("data", (data: Buffer) => {
        const output = data.toString()
        const match = output.match(/listening on (https?:\/\/[^\s]+)/)
        if (match && !resolved) {
          serverUrl = match[1]
          appState.openCodeUrl = serverUrl
          resolved = true
          resolve(serverUrl)
        }
      })

      openCodeProcess.on("error", (err) => {
        // opencode not installed or other error
        if (!resolved) {
          resolved = true
          resolve(null)
        }
      })

      openCodeProcess.on("exit", () => {
        openCodeProcess = null
        serverUrl = null
        appState.openCodeUrl = null
      })

      // Register cleanup to kill process on shutdown
      registerCleanup("opencode-server", () => {
        stopOpenCodeServer()
      })

      // Timeout after 10 seconds if no URL found
      setTimeout(() => {
        if (!resolved) {
          resolved = true
          resolve(null)
        }
      }, 10000)

    } catch {
      resolve(null)
    }
  })
}

/**
 * Stop the OpenCode server.
 */
export function stopOpenCodeServer(): void {
  if (openCodeProcess && !openCodeProcess.killed) {
    openCodeProcess.kill("SIGTERM")
    openCodeProcess = null
    serverUrl = null
    appState.openCodeUrl = null
  }
}

/**
 * Check if OpenCode server is running.
 */
export function isOpenCodeServerRunning(): boolean {
  return openCodeProcess !== null && !openCodeProcess.killed
}

/**
 * Get the current server URL.
 */
export function getOpenCodeServerUrl(): string | null {
  return serverUrl
}
