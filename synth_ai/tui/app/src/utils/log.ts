/**
 * Logging utility - writes to ~/.synth-ai/tui/logs/{prefix}_{timestamp}.log
 * Each prefix gets a unique session-based timestamp file.
 */
import fs from "node:fs"
import path from "node:path"
import { tuiLogsDir } from "../paths"

// Cache log file paths per prefix so all logs in a session go to the same file
const logFilePaths = new Map<string, string>()

function ensureLogDir(): void {
  try {
    fs.mkdirSync(tuiLogsDir, { recursive: true })
  } catch {
    // Best-effort; logging shouldn't crash the app.
  }
}

function getLogFilePath(prefix?: string): string {
  const key = prefix ?? ""
  if (logFilePaths.has(key)) {
    return logFilePaths.get(key)!
  }

  // Create timestamp for this session: YYYY-MM-DD_HH-MM-SS-mmm
  const now = new Date()
  const timestamp = now.toISOString().replace(/[:.]/g, "-").replace("T", "_").replace("Z", "")
  const filename = prefix ? `${prefix}_${timestamp}.log` : `${timestamp}.log`
  const filePath = path.join(tuiLogsDir, filename)

  logFilePaths.set(key, filePath)
  return filePath
}

export function log(prefix: string | undefined, ...args: any[]): void {
  const ts = new Date().toISOString()
  const msg = args
    .map((a) => (typeof a === "object" ? JSON.stringify(a, null, 2) : String(a)))
    .join(" ")
  ensureLogDir()
  try {
    fs.appendFileSync(getLogFilePath(prefix), `[${ts}] ${msg}\n`)
  } catch {
    // Ignore logging errors.
  }
}
