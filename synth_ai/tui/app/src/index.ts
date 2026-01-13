/**
 * synth-ai TUI - Entry point
 *
 * This is the thin entrypoint for the TUI application.
 * All logic is in app.ts and its dependencies.
 */
import { shutdown, registerCleanup } from "./lifecycle"
import { runSolidApp } from "./solid/app"
import { initLogger, cleanupLogger } from "./services"
import { initModeState, getCurrentMode } from "./state/mode"

// Initialize mode state first (determines URLs based on env)
initModeState()

// Initialize logger early (before any output)
initLogger(getCurrentMode())
registerCleanup("tui-logger", cleanupLogger)

// Log but don't crash - TUI should survive backend issues
process.on("unhandledRejection", (err) => {
  process.stderr.write(`Unhandled rejection: ${err}\n`)
})
process.on("uncaughtException", (err) => {
  process.stderr.write(`Uncaught exception: ${err}\n`)
})

runSolidApp().catch(() => {
  // Fatal startup error - clean exit
  void shutdown(1)
})
