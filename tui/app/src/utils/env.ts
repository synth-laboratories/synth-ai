/**
 * Environment-style parsing utilities for local settings.
 */

export function parseEnvFile(content: string): Record<string, string> {
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

export function formatEnvLine(key: string, value: string): string {
  return `${key}=${escapeEnvValue(value)}`
}

export function escapeEnvValue(value: string): string {
  const safe = value ?? ""
  return `"${safe.replace(/\\/g, "\\\\").replace(/\"/g, '\\"')}"`
}

/** Stub - env key scanning was removed. Returns empty array. */
export async function scanEnvKeys(_scanRoot?: string): Promise<Array<{ key: string; sources: string[]; varNames: string[] }>> {
  return []
}
