/**
 * Centralized shutdown manager for clean app termination.
 *
 * Provides a single point of control for all cleanup operations:
 * - Aborts in-flight fetch requests via AbortController
 * - Clears all registered intervals and timeouts
 * - Runs registered cleanup functions (SSE disconnect, etc.)
 * - Restores terminal state with explicit ANSI sequences
 * - Handles SIGINT/SIGTERM gracefully
 */

export type CleanupFn = () => void | Promise<void>

// ANSI sequences for terminal restoration
const ANSI_RESET = "\x1b[0m" // Reset all attributes
const ANSI_SHOW_CURSOR = "\x1b[?25h" // Show cursor
const ANSI_EXIT_ALT_SCREEN = "\x1b[?1049l" // Exit alternate screen buffer

interface ShutdownState {
  abortController: AbortController
  intervals: Set<ReturnType<typeof setInterval>>
  timeouts: Set<ReturnType<typeof setTimeout>>
  cleanups: Map<string, CleanupFn>
  isShuttingDown: boolean
  renderer: { stop: () => void; destroy: () => void } | null
}

const state: ShutdownState = {
  abortController: new AbortController(),
  intervals: new Set(),
  timeouts: new Set(),
  cleanups: new Map(),
  isShuttingDown: false,
  renderer: null,
}

/**
 * Get the global abort signal for fetch requests.
 * Pass this to fetch calls to allow cancellation on shutdown.
 */
export function getAbortSignal(): AbortSignal {
  return state.abortController.signal
}

/**
 * Check if shutdown is in progress.
 */
export function isShuttingDown(): boolean {
  return state.isShuttingDown
}

/**
 * Register the renderer for cleanup.
 */
export function registerRenderer(renderer: { stop: () => void; destroy: () => void }): void {
  state.renderer = renderer
}

/**
 * Register an interval for cleanup. Returns the interval ID for convenience.
 */
export function registerInterval(id: ReturnType<typeof setInterval>): ReturnType<typeof setInterval> {
  state.intervals.add(id)
  return id
}

/**
 * Register a timeout for cleanup. Returns the timeout ID for convenience.
 */
export function registerTimeout(id: ReturnType<typeof setTimeout>): ReturnType<typeof setTimeout> {
  state.timeouts.add(id)
  return id
}

/**
 * Unregister a timeout (e.g., when it fires or is manually cleared).
 */
export function unregisterTimeout(id: ReturnType<typeof setTimeout>): void {
  state.timeouts.delete(id)
}

/**
 * Register a named cleanup function.
 */
export function registerCleanup(name: string, fn: CleanupFn): void {
  state.cleanups.set(name, fn)
}

/**
 * Unregister a cleanup function by name.
 */
export function unregisterCleanup(name: string): void {
  state.cleanups.delete(name)
}

/**
 * Main shutdown function - idempotent, safe to call multiple times.
 * Only the first call executes; subsequent calls block forever.
 */
export async function shutdown(exitCode: number = 0): Promise<never> {
  // Prevent re-entrant shutdown
  if (state.isShuttingDown) {
    return new Promise(() => {}) as never // Block forever, first shutdown will exit
  }
  state.isShuttingDown = true

  // 1. Abort all in-flight fetch requests
  state.abortController.abort()

  // 2. Clear all intervals
  for (const id of state.intervals) {
    clearInterval(id)
  }
  state.intervals.clear()

  // 3. Clear all timeouts
  for (const id of state.timeouts) {
    clearTimeout(id)
  }
  state.timeouts.clear()

  // 4. Run registered cleanup functions
  for (const [, fn] of state.cleanups) {
    try {
      await fn()
    } catch {
      // Swallow errors during shutdown
    }
  }
  state.cleanups.clear()

  // 5. Stop and destroy renderer
  if (state.renderer) {
    try {
      state.renderer.stop()
      state.renderer.destroy()
    } catch {
      // Swallow errors
    }
  }

  // 6. Force terminal restoration (belt and suspenders)
  process.stdout.write(ANSI_SHOW_CURSOR)
  process.stdout.write(ANSI_EXIT_ALT_SCREEN)
  process.stdout.write(ANSI_RESET)
  process.stdout.write("\n")

  // 7. Exit
  process.exit(exitCode)
}

/**
 * Install process signal handlers. Call once at app startup.
 */
export function installSignalHandlers(): void {
  process.on("SIGINT", () => void shutdown(0))
  process.on("SIGTERM", () => void shutdown(0))
}
