/**
 * synth-ai TUI - Entry point
 *
 * This is the thin entrypoint for the TUI application.
 * All logic is in app.ts and its dependencies.
 */
import { runApp } from "./app"

runApp().catch((err) => {
  console.error("Fatal error:", err)
  process.exit(1)
})
