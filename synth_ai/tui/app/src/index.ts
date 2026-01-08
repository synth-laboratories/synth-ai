/**
 * synth-ai TUI - Entry point
 *
 * This is the thin entrypoint for the TUI application.
 * All logic is in app.ts and its dependencies.
 *
 * IMPORTANT: We load .env files synchronously BEFORE importing any other modules
 * so that process.env is populated with SYNTH_API_KEY for local/dev backends.
 */
import { runApp } from "./app"
import { shutdown } from "./lifecycle"

// Swallow unhandled errors - TUI should survive backend issues
process.on("unhandledRejection", () => {
  // Ignore - don't crash or exit
})
process.on("uncaughtException", () => {
  // Ignore - don't crash or exit
})

runApp().catch(() => {
  // Fatal startup error - clean exit
  void shutdown(1)
})
