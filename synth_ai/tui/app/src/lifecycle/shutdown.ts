/**
 * Centralized shutdown manager for clean app termination.
 *
 * Provides a single point of control for all cleanup operations:
 * - Aborts in-flight fetch requests via AbortController
 * - Runs registered cleanup functions (intervals, SSE disconnect, etc.)
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
  cleanups: Map<string, CleanupFn>
  isShuttingDown: boolean
  renderer: { stop: () => void; destroy: () => void } | null
}

const state: ShutdownState = {
  abortController: new AbortController(),
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

  // 2. Run registered cleanup functions
  for (const [, fn] of state.cleanups) {
    try {
      await fn()
    } catch {
      // Swallow errors during shutdown
    }
  }
  state.cleanups.clear()

  // 3. Stop and destroy renderer
  if (state.renderer) {
    try {
      state.renderer.stop()
      state.renderer.destroy()
    } catch {
      // Swallow errors
    }
  }

  // 4. Force terminal restoration (belt and suspenders)
  process.stdout.write(ANSI_SHOW_CURSOR)
  process.stdout.write(ANSI_EXIT_ALT_SCREEN)
  process.stdout.write(ANSI_RESET)
  process.stdout.write("\n")

  // 5. Exit
  process.exit(exitCode)
}

/**
 * Install process signal handlers. Call once at app startup.
 */
export function installSignalHandlers(): void {
  process.on("SIGINT", () => void shutdown(0))
  process.on("SIGTERM", () => void shutdown(0))
}
