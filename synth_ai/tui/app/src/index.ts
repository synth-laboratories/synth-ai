/**
 * synth-ai TUI - Entry point
 *
 * This is the thin entrypoint for the TUI application.
 * All logic is in app.ts and its dependencies.
 *
 * IMPORTANT: We load .env files synchronously BEFORE importing any other modules
 * so that process.env is populated with SYNTH_API_KEY for local/dev backends.
 */
import { shutdown } from "./lifecycle"
import { runSolidApp } from "./solid/app"

// Swallow unhandled errors - TUI should survive backend issues
process.on("unhandledRejection", () => {
  // Ignore - don't crash or exit
})
process.on("uncaughtException", () => {
  // Ignore - don't crash or exit
})

runSolidApp().catch(() => {
  // Fatal startup error - clean exit
  void shutdown(1)
})
