/**
 * synth-ai TUI - Entry point
 *
 * This is the thin entrypoint for the TUI application.
 * All logic is in app.ts and its dependencies.
 */
import { shutdown } from "./lifecycle"
import { runSolidApp } from "./solid/app"

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
