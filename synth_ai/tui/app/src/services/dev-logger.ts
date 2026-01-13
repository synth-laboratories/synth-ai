/**
 * TUI logger - redirects all stdio to timestamped log files.
 *
 * All console output and process stdio is captured and written to
 * ~/.synth-ai/tui/logs/{timestamp}.log
 */

import * as fs from "fs"
import * as path from "path"
import type { Mode } from "../types"
import { tuiLogsDir } from "../paths"

interface LoggerState {
  initialized: boolean
  logFilePath: string | null
  writeStream: fs.WriteStream | null
  originalStdoutWrite: typeof process.stdout.write | null
  originalStderrWrite: typeof process.stderr.write | null
  originalConsoleLog: typeof console.log | null
  originalConsoleError: typeof console.error | null
  originalConsoleWarn: typeof console.warn | null
  originalConsoleInfo: typeof console.info | null
  originalConsoleDebug: typeof console.debug | null
}

const state: LoggerState = {
  initialized: false,
  logFilePath: null,
  writeStream: null,
  originalStdoutWrite: null,
  originalStderrWrite: null,
  originalConsoleLog: null,
  originalConsoleError: null,
  originalConsoleWarn: null,
  originalConsoleInfo: null,
  originalConsoleDebug: null,
}

/**
 * Get the logs directory path.
 */
function getLogsDir(): string {
  return tuiLogsDir
}

/**
 * Generate a timestamped filename.
 */
function generateLogFilename(): string {
  const now = new Date()
  const timestamp = now.toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, -5)
  return `${timestamp}.log`
}

/**
 * Ensure the logs directory exists.
 */
function ensureLogsDir(): void {
  const dir = getLogsDir()
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }
}

/**
 * Format a message for logging with timestamp and level.
 */
function formatLogMessage(level: string, args: unknown[]): string {
  const ts = new Date().toISOString()
  const msg = args
    .map((a) => {
      if (a === undefined) return "undefined"
      if (a === null) return "null"
      if (typeof a === "object") {
        try {
          return JSON.stringify(a, null, 2)
        } catch {
          return String(a)
        }
      }
      return String(a)
    })
    .join(" ")
  return `[${ts}] [${level}] ${msg}\n`
}

/**
 * Write to the log file if initialized.
 */
function writeToLog(data: string | Uint8Array): void {
  if (state.writeStream && !state.writeStream.destroyed) {
    const text = typeof data === "string" ? data : Buffer.from(data).toString("utf-8")
    state.writeStream.write(text)
  }
}

/**
 * Initialize the TUI logger.
 *
 * Call this early in app startup before any output.
 * Returns the log file path if logging was enabled, null otherwise.
 */
export function initLogger(mode: Mode): string | null {
  if (state.initialized) {
    return state.logFilePath
  }

  try {
    ensureLogsDir()

    const logDir = getLogsDir()
    const filename = generateLogFilename()
    state.logFilePath = path.join(logDir, filename)

    // Create write stream
    state.writeStream = fs.createWriteStream(state.logFilePath, { flags: "a" })

    // Write header
    state.writeStream.write(`\n${"=".repeat(60)}\n`)
    state.writeStream.write(`TUI Log - Mode: ${mode}\n`)
    state.writeStream.write(`Started: ${new Date().toISOString()}\n`)
    state.writeStream.write(`${"=".repeat(60)}\n\n`)

    // Save original methods (capture in local vars for closure safety)
    const origStdoutWrite = process.stdout.write.bind(process.stdout)
    const origStderrWrite = process.stderr.write.bind(process.stderr)
    const origConsoleLog = console.log.bind(console)
    const origConsoleError = console.error.bind(console)
    const origConsoleWarn = console.warn.bind(console)
    const origConsoleInfo = console.info.bind(console)
    const origConsoleDebug = console.debug.bind(console)

    state.originalStdoutWrite = origStdoutWrite
    state.originalStderrWrite = origStderrWrite
    state.originalConsoleLog = origConsoleLog
    state.originalConsoleError = origConsoleError
    state.originalConsoleWarn = origConsoleWarn
    state.originalConsoleInfo = origConsoleInfo
    state.originalConsoleDebug = origConsoleDebug

    // Override process.stdout.write
    process.stdout.write = function (
      data: string | Uint8Array,
      encodingOrCallback?: BufferEncoding | ((err?: Error) => void),
      callback?: (err?: Error) => void
    ): boolean {
      writeToLog(data)
      if (typeof encodingOrCallback === "function") {
        return origStdoutWrite(data, encodingOrCallback)
      }
      return origStdoutWrite(data, encodingOrCallback, callback)
    } as typeof process.stdout.write

    // Override process.stderr.write
    process.stderr.write = function (
      data: string | Uint8Array,
      encodingOrCallback?: BufferEncoding | ((err?: Error) => void),
      callback?: (err?: Error) => void
    ): boolean {
      writeToLog(data)
      if (typeof encodingOrCallback === "function") {
        return origStderrWrite(data, encodingOrCallback)
      }
      return origStderrWrite(data, encodingOrCallback, callback)
    } as typeof process.stderr.write

    // Override console methods
    console.log = (...args: unknown[]) => {
      writeToLog(formatLogMessage("LOG", args))
      origConsoleLog(...args)
    }

    console.error = (...args: unknown[]) => {
      writeToLog(formatLogMessage("ERROR", args))
      origConsoleError(...args)
    }

    console.warn = (...args: unknown[]) => {
      writeToLog(formatLogMessage("WARN", args))
      origConsoleWarn(...args)
    }

    console.info = (...args: unknown[]) => {
      writeToLog(formatLogMessage("INFO", args))
      origConsoleInfo(...args)
    }

    console.debug = (...args: unknown[]) => {
      writeToLog(formatLogMessage("DEBUG", args))
      origConsoleDebug(...args)
    }

    state.initialized = true
    return state.logFilePath
  } catch {
    // Silently fail - logging shouldn't break the app
    return null
  }
}

/**
 * Get the current log file path if logging is enabled.
 */
export function getLogPath(): string | null {
  return state.logFilePath
}

/**
 * Check if logging is currently active.
 */
export function isLoggingActive(): boolean {
  return state.initialized && state.writeStream !== null
}

/**
 * Write a message directly to the log (bypasses console override).
 */
export function tuiLog(level: "LOG" | "ERROR" | "WARN" | "INFO" | "DEBUG", ...args: unknown[]): void {
  if (!state.initialized || !state.writeStream) return
  writeToLog(formatLogMessage(level, args))
}

/**
 * Cleanup the logger - close streams and restore original methods.
 * Call this during app shutdown.
 */
export function cleanupLogger(): void {
  if (!state.initialized) return

  // Write footer
  if (state.writeStream && !state.writeStream.destroyed) {
    state.writeStream.write(`\n${"=".repeat(60)}\n`)
    state.writeStream.write(`TUI Log Ended: ${new Date().toISOString()}\n`)
    state.writeStream.write(`${"=".repeat(60)}\n`)
  }

  // Restore original methods
  if (state.originalStdoutWrite) {
    process.stdout.write = state.originalStdoutWrite
  }
  if (state.originalStderrWrite) {
    process.stderr.write = state.originalStderrWrite
  }
  if (state.originalConsoleLog) {
    console.log = state.originalConsoleLog
  }
  if (state.originalConsoleError) {
    console.error = state.originalConsoleError
  }
  if (state.originalConsoleWarn) {
    console.warn = state.originalConsoleWarn
  }
  if (state.originalConsoleInfo) {
    console.info = state.originalConsoleInfo
  }
  if (state.originalConsoleDebug) {
    console.debug = state.originalConsoleDebug
  }

  // Close write stream
  if (state.writeStream && !state.writeStream.destroyed) {
    state.writeStream.end()
  }

  // Reset state
  state.initialized = false
  state.logFilePath = null
  state.writeStream = null
  state.originalStdoutWrite = null
  state.originalStderrWrite = null
  state.originalConsoleLog = null
  state.originalConsoleError = null
  state.originalConsoleWarn = null
  state.originalConsoleInfo = null
  state.originalConsoleDebug = null
}
