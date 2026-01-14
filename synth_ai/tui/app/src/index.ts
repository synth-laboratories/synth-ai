/**
 * synth-ai TUI - Entry point
 *
 * This is the thin entrypoint for the TUI application.
 * All logic is in app.ts and its dependencies.
 */
import { installSignalHandlers, isShuttingDown, registerCleanup, shutdown } from "./lifecycle"
import { runSolidApp } from "./solid/app"
import { initLogger, cleanupLogger } from "./services"
import { initModeState, getCurrentMode } from "./state/mode"

// Initialize mode state first (determines URLs based on env)
initModeState()

// Initialize logger early (before any output)
initLogger(getCurrentMode())
registerCleanup("tui-logger", cleanupLogger)

installSignalHandlers()

function formatError(err: unknown): string {
  if (err instanceof Error) {
    return err.stack || err.message
  }
  return String(err)
}

function handleFatalError(label: string, err: unknown): void {
  process.stderr.write(`${label}: ${formatError(err)}\n`)
  if (!isShuttingDown()) {
    void shutdown(1)
  }
}

process.on("unhandledRejection", (err) => {
  handleFatalError("Unhandled rejection", err)
})
process.on("uncaughtException", (err) => {
  handleFatalError("Uncaught exception", err)
})

runSolidApp().catch((err) => {
  handleFatalError("Fatal startup error", err)
})
