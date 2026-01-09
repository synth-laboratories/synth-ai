/**
 * OpenCode server management - auto-start and lifecycle management.
 */
import { spawn, type ChildProcess } from "child_process"
import fs from "fs"
import path from "path"
import { registerCleanup } from "../lifecycle"
import { appState } from "../state/app-state"

let openCodeProcess: ChildProcess | null = null
let serverUrl: string | null = null
const DEFAULT_STARTUP_TIMEOUT_MS = 60000

type OpenCodeLaunch = {
  command: string
  args: string[]
  cwd?: string
}

function resolveBunCommand(): string {
  return process.env.OPENCODE_BUN_PATH || "bun"
}

function resolveLocalOpenCode(): OpenCodeLaunch | null {
  const envRoot =
    process.env.OPENCODE_DEV_PATH ||
    process.env.OPENCODE_DEV_ROOT ||
    process.env.OPENCODE_PATH
  const candidates = [envRoot].filter(Boolean) as string[]
  const allowAutoLocal = process.env.OPENCODE_USE_LOCAL === "1"
  if (allowAutoLocal) {
    candidates.push(path.resolve(__dirname, "../../../../..", "..", "opencode"))
  }

  if (candidates.length === 0) {
    return null
  }

  for (const candidate of candidates) {
    const entry = path.join(candidate, "packages", "opencode", "src", "index.ts")
    if (fs.existsSync(entry)) {
      const tsconfig = path.join(candidate, "packages", "opencode", "tsconfig.json")
      const args = ["--preload", "@opentui/solid/preload"]
      if (fs.existsSync(tsconfig)) {
        args.push("--tsconfig-override", tsconfig)
      }
      args.push(entry, "serve")
      return {
        command: resolveBunCommand(),
        args,
        cwd: candidate,
      }
    }
  }

  return null
}

function resolveStartupTimeoutMs(): number {
  const raw = process.env.OPENCODE_STARTUP_TIMEOUT_MS
  if (!raw) return DEFAULT_STARTUP_TIMEOUT_MS
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_STARTUP_TIMEOUT_MS
}

function findInPath(command: string): string | null {
  const envPath = process.env.PATH
  if (!envPath) return null
  for (const entry of envPath.split(path.delimiter)) {
    if (!entry) continue
    const candidate = path.join(entry, command)
    if (fs.existsSync(candidate)) {
      return candidate
    }
  }
  return null
}

function resolveOpenCodeCommand(): string | null {
  const override = process.env.OPENCODE_CMD
  if (override) return override
  const pathCommand = findInPath("opencode-synth") ?? findInPath("opencode")
  if (pathCommand) return pathCommand
  const fallbackCandidates = [
    "/opt/homebrew/bin/opencode-synth",
    "/usr/local/bin/opencode-synth",
    "/opt/homebrew/bin/opencode",
    "/usr/local/bin/opencode",
  ]
  for (const candidate of fallbackCandidates) {
    if (fs.existsSync(candidate)) return candidate
  }
  return null
}

/**
 * Start the OpenCode server in the background.
 * Returns the server URL once it's ready.
 */
export async function startOpenCodeServer(): Promise<string | null> {
  // Don't start if already running
  if (openCodeProcess && !openCodeProcess.killed) {
    return serverUrl
  }

  const localLaunch = resolveLocalOpenCode()
  const fallbackCommand = resolveOpenCodeCommand()
  const launch: OpenCodeLaunch | null = localLaunch ?? (fallbackCommand ? {
    command: fallbackCommand,
    args: ["serve"],
  } : null)

  if (!launch) {
    return null
  }

  // Check if user has opencode installed or a local dev checkout
  return new Promise((resolve) => {
    try {
      openCodeProcess = spawn(launch.command, launch.args, {
        stdio: ["ignore", "pipe", "pipe"],
        detached: false,
        cwd: launch.cwd,
      })

      let resolved = false
      let hasUrl = false

      // Parse stdout for the server URL
      openCodeProcess.stdout?.on("data", (data: Buffer) => {
        const output = data.toString()
        // Look for "listening on http://..." pattern
        const match = output.match(/listening on (https?:\/\/[^\s]+)/)
        if (match) {
          if (!hasUrl) {
            serverUrl = match[1]
            appState.openCodeUrl = serverUrl
            hasUrl = true
          }
          if (!resolved) {
            resolved = true
            resolve(serverUrl)
          }
        }
      })

      // Also check stderr (some tools output there)
      openCodeProcess.stderr?.on("data", (data: Buffer) => {
        const output = data.toString()
        const match = output.match(/listening on (https?:\/\/[^\s]+)/)
        if (match) {
          if (!hasUrl) {
            serverUrl = match[1]
            appState.openCodeUrl = serverUrl
            hasUrl = true
          }
          if (!resolved) {
            resolved = true
            resolve(serverUrl)
          }
        }
      })

      openCodeProcess.on("error", (_err) => {
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
        if (!resolved) {
          resolved = true
          resolve(null)
        }
      })

      // Register cleanup to kill process on shutdown
      registerCleanup("opencode-server", () => {
        stopOpenCodeServer()
      })

      // Timeout after a longer window for first-run installs.
      const timeoutMs = resolveStartupTimeoutMs()
      setTimeout(() => {
        if (!resolved) {
          resolved = true
          resolve(null)
        }
      }, timeoutMs)

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
