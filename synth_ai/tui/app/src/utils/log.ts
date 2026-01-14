/**
 * Unified logging utility - writes to ~/.synth-ai/tui/logs/
 *
 * Two logging systems in one:
 * 1. Structured category logging via log() - writes to {timestamp}_{category}.log
 * 2. Console capture via initLogger() - captures all stdout/stderr to {timestamp}_console.log
 */
import fs from "node:fs"
import path from "node:path"
import type { Mode } from "../types"
import { tuiLogsDir } from "../paths"

// ============================================================================
// Shared state
// ============================================================================

const logFilePaths = new Map<string, string>()
let sessionTimestamp: string | null = null

function ensureLogDir(): void {
  try {
    fs.mkdirSync(tuiLogsDir, { recursive: true })
  } catch {
    // Best-effort; logging shouldn't crash the app.
  }
}

function getSessionTimestamp(): string {
  if (sessionTimestamp) return sessionTimestamp
  const now = new Date()
  sessionTimestamp = now.toISOString().replace(/[:.]/g, "-").replace("T", "_").replace("Z", "")
  return sessionTimestamp
}

function getLogFilePath(suffix: string): string {
  if (logFilePaths.has(suffix)) return logFilePaths.get(suffix)!

  const filePath = path.join(tuiLogsDir, `${getSessionTimestamp()}_${suffix}.log`)
  logFilePaths.set(suffix, filePath)
  return filePath
}

// ============================================================================
// Structured category logging
// ============================================================================

function stringify(value: unknown): string {
  if (typeof value === "string") return value
  if (value == null) return String(value)
  if (typeof value !== "object") return String(value)
  try {
    const seen = new WeakSet<object>()
    return JSON.stringify(
      value,
      (_key, next) => {
        if (typeof next === "bigint") return next.toString()
        if (typeof next === "object" && next !== null) {
          if (seen.has(next)) return "[Circular]"
          seen.add(next)
        }
        return next
      },
      2,
    )
  } catch {
    return Object.prototype.toString.call(value)
  }
}

export type LogCategory = "key" | "modal" | "state" | "action" | "error" | "lifecycle" | "http"

function formatCategory(category: LogCategory): string {
  return `[${category.toUpperCase().padEnd(9)}]`
}

function writeLog(filePath: string, line: string): void {
  try {
    fs.appendFileSync(filePath, line)
  } catch {
    // Ignore logging errors.
  }
}

/**
 * Log a message with category.
 * Writes to both main log (with category prefix) and category-specific log (without prefix).
 */
export function log(category: LogCategory, message: string, data?: unknown): void {
  try {
    ensureLogDir()
    const ts = new Date().toISOString()
    const content = data !== undefined ? `${message} ${stringify(data)}` : message

    // All log includes category prefix for filtering/context
    const allLine = `[${ts}] ${formatCategory(category)} ${content}\n`
    writeLog(getLogFilePath("all"), allLine)

    // Category-specific log omits redundant category prefix
    const categoryLine = `[${ts}] ${content}\n`
    writeLog(getLogFilePath(category), categoryLine)
  } catch {
    // Ignore logging errors.
  }
}

/**
 * Format and log a key event.
 */
export function logKey(event: { name?: string; sequence?: string; ctrl?: boolean; meta?: boolean; shift?: boolean }, context?: string): void {
  const parts: string[] = []
  if (event.ctrl) parts.push("ctrl")
  if (event.meta) parts.push("meta")
  if (event.shift) parts.push("shift")
  if (event.name) parts.push(event.name)
  else if (event.sequence) parts.push(`seq:${JSON.stringify(event.sequence)}`)

  const keyStr = parts.join("+") || "unknown"
  const contextStr = context ? ` (context: ${context})` : ""
  log("key", `pressed: ${keyStr}${contextStr}`)
}

/**
 * Log an error with stack trace extraction.
 */
export function logError(message: string, error: unknown): void {
  const errorInfo = error instanceof Error
    ? { message: error.message, stack: error.stack }
    : error
  log("error", message, errorInfo)
}

let globalHandlersInstalled = false

/**
 * Install global error handlers for uncaught exceptions and unhandled rejections.
 */
export function installGlobalErrorHandlers(): void {
  if (globalHandlersInstalled) return
  globalHandlersInstalled = true

  process.on("uncaughtException", (err) => {
    logError("uncaughtException", err)
  })

  process.on("unhandledRejection", (reason) => {
    logError("unhandledRejection", reason)
  })

  log("lifecycle", "global error handlers installed")
}

// ============================================================================
// Console capture (safety net for all stdout/stderr)
// ============================================================================

interface ConsoleLoggerState {
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

const consoleState: ConsoleLoggerState = {
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

function formatConsoleMessage(level: string, args: unknown[]): string {
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

function writeToConsoleLog(data: string | Uint8Array): void {
  if (consoleState.writeStream && !consoleState.writeStream.destroyed) {
    const text = typeof data === "string" ? data : Buffer.from(data).toString("utf-8")
    consoleState.writeStream.write(text)
  }
}

/**
 * Initialize the console capture logger.
 * Call this early in app startup before any output.
 * Returns the log file path if logging was enabled, null otherwise.
 */
export function initLogger(_mode: Mode): string | null {
  if (consoleState.initialized) {
    return consoleState.logFilePath
  }

  try {
    ensureLogDir()
    consoleState.logFilePath = getLogFilePath("console")

    // Create write stream
    consoleState.writeStream = fs.createWriteStream(consoleState.logFilePath, { flags: "a" })

    // Save original methods (capture in local vars for closure safety)
    const origStdoutWrite = process.stdout.write.bind(process.stdout)
    const origStderrWrite = process.stderr.write.bind(process.stderr)
    const origConsoleLog = console.log.bind(console)
    const origConsoleError = console.error.bind(console)
    const origConsoleWarn = console.warn.bind(console)
    const origConsoleInfo = console.info.bind(console)
    const origConsoleDebug = console.debug.bind(console)

    consoleState.originalStdoutWrite = origStdoutWrite
    consoleState.originalStderrWrite = origStderrWrite
    consoleState.originalConsoleLog = origConsoleLog
    consoleState.originalConsoleError = origConsoleError
    consoleState.originalConsoleWarn = origConsoleWarn
    consoleState.originalConsoleInfo = origConsoleInfo
    consoleState.originalConsoleDebug = origConsoleDebug

    // Override process.stdout.write
    process.stdout.write = function (
      data: string | Uint8Array,
      encodingOrCallback?: BufferEncoding | ((err?: Error) => void),
      callback?: (err?: Error) => void
    ): boolean {
      writeToConsoleLog(data)
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
      writeToConsoleLog(data)
      if (typeof encodingOrCallback === "function") {
        return origStderrWrite(data, encodingOrCallback)
      }
      return origStderrWrite(data, encodingOrCallback, callback)
    } as typeof process.stderr.write

    // Override console methods
    console.log = (...args: unknown[]) => {
      writeToConsoleLog(formatConsoleMessage("LOG", args))
      origConsoleLog(...args)
    }

    console.error = (...args: unknown[]) => {
      writeToConsoleLog(formatConsoleMessage("ERROR", args))
      origConsoleError(...args)
    }

    console.warn = (...args: unknown[]) => {
      writeToConsoleLog(formatConsoleMessage("WARN", args))
      origConsoleWarn(...args)
    }

    console.info = (...args: unknown[]) => {
      writeToConsoleLog(formatConsoleMessage("INFO", args))
      origConsoleInfo(...args)
    }

    console.debug = (...args: unknown[]) => {
      writeToConsoleLog(formatConsoleMessage("DEBUG", args))
      origConsoleDebug(...args)
    }

    consoleState.initialized = true
    return consoleState.logFilePath
  } catch {
    // Silently fail - logging shouldn't break the app
    return null
  }
}

/**
 * Get the current console log file path if logging is enabled.
 */
export function getLogPath(): string | null {
  return consoleState.logFilePath
}

/**
 * Check if console logging is currently active.
 */
export function isLoggingActive(): boolean {
  return consoleState.initialized && consoleState.writeStream !== null
}

/**
 * Write a message directly to the console log (bypasses console override).
 */
export function tuiLog(level: "LOG" | "ERROR" | "WARN" | "INFO" | "DEBUG", ...args: unknown[]): void {
  if (!consoleState.initialized || !consoleState.writeStream) return
  writeToConsoleLog(formatConsoleMessage(level, args))
}

/**
 * Cleanup the console logger - close streams and restore original methods.
 * Call this during app shutdown.
 */
export function cleanupLogger(): void {
  if (!consoleState.initialized) return

  // Restore original methods
  if (consoleState.originalStdoutWrite) {
    process.stdout.write = consoleState.originalStdoutWrite
  }
  if (consoleState.originalStderrWrite) {
    process.stderr.write = consoleState.originalStderrWrite
  }
  if (consoleState.originalConsoleLog) {
    console.log = consoleState.originalConsoleLog
  }
  if (consoleState.originalConsoleError) {
    console.error = consoleState.originalConsoleError
  }
  if (consoleState.originalConsoleWarn) {
    console.warn = consoleState.originalConsoleWarn
  }
  if (consoleState.originalConsoleInfo) {
    console.info = consoleState.originalConsoleInfo
  }
  if (consoleState.originalConsoleDebug) {
    console.debug = consoleState.originalConsoleDebug
  }

  // Close write stream
  if (consoleState.writeStream && !consoleState.writeStream.destroyed) {
    consoleState.writeStream.end()
  }

  // Reset state
  consoleState.initialized = false
  consoleState.logFilePath = null
  consoleState.writeStream = null
  consoleState.originalStdoutWrite = null
  consoleState.originalStderrWrite = null
  consoleState.originalConsoleLog = null
  consoleState.originalConsoleError = null
  consoleState.originalConsoleWarn = null
  consoleState.originalConsoleInfo = null
  consoleState.originalConsoleDebug = null
}
