/**
 * synth-ai TUI - Entry point
 *
 * This is the thin entrypoint for the TUI application.
 * All logic is in app.ts and its dependencies.
 *
 * IMPORTANT: We load .env files synchronously BEFORE importing any other modules
 * so that process.env is populated with SYNTH_API_KEY for local/dev backends.
 */
import { readFileSync, existsSync } from "node:fs"
import { join } from "node:path"

/**
 * Parse a .env file content and return key-value pairs.
 * Handles quoted values and inline comments.
 */
function parseEnvContent(content: string): Record<string, string> {
  const values: Record<string, string> = {}
  const lines = content.split(/\r?\n/)
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#")) continue
    const match = trimmed.match(/^(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.+)$/)
    if (!match) continue
    const key = match[1]
    let value = match[2].trim()
    if (
      (value.startsWith("\"") && value.endsWith("\"")) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      const quoted = value
      value = value.slice(1, -1)
      if (quoted.startsWith("\"")) {
        value = value.replace(/\\\\/g, "\\").replace(/\\"/g, "\"")
      }
    } else {
      value = value.split(/\s+#/)[0].trim()
    }
    values[key] = value
  }
  return values
}

/**
 * Load .env files from the current directory and parent directories.
 * Only sets variables that are not already defined in process.env.
 */
function loadEnvFiles(): void {
  const cwd = process.cwd()
  const envFiles = [
    join(cwd, ".env"),
    join(cwd, ".env.local"),
    join(cwd, "backend", ".env"),  // Common monorepo structure
  ]

  // Also check parent directory for monorepo setups
  const parentDir = join(cwd, "..")
  envFiles.push(join(parentDir, ".env"))

  for (const envFile of envFiles) {
    try {
      if (!existsSync(envFile)) continue
      const content = readFileSync(envFile, "utf8")
      const values = parseEnvContent(content)
      for (const [key, value] of Object.entries(values)) {
        // Only set if not already defined (environment takes precedence)
        if (process.env[key] === undefined || process.env[key] === "") {
          process.env[key] = value
        }
      }
    } catch {
      // Ignore read errors
    }
  }
}

// Load .env files before importing app modules
loadEnvFiles()

// Now import and run the app (after process.env is populated)
import("./app").then(({ runApp }) => {
  runApp().catch((err) => {
    console.error("Fatal error:", err)
    process.exit(1)
  })
})
