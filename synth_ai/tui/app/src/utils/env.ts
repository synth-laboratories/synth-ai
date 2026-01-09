/**
 * Environment file parsing and scanning utilities.
 */

import path from "node:path"
import { promises as fs } from "node:fs"
import type { EnvKeyOption } from "../types"
import { config } from "../state/polling"

const IGNORED_DIRS = new Set([
  ".git",
  "node_modules",
  ".venv",
  "venv",
  "__pycache__",
  ".pytest_cache",
  ".mypy_cache",
  ".ruff_cache",
  ".tox",
  ".next",
  ".turbo",
  ".cache",
  "dist",
  "build",
  "out",
])

const KEY_VAR_NAMES = new Set([
  "SYNTH_API_KEY",
  "SYNTH_TUI_API_KEY_PROD",
  "SYNTH_TUI_API_KEY_DEV",
  "SYNTH_TUI_API_KEY_LOCAL",
])

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

export function parseEnvKeys(
  content: string,
  sourcePath: string,
  scanRoot: string = config.envKeyScanRoot,
): Array<{ key: string; source: string; varName: string }> {
  const results: Array<{ key: string; source: string; varName: string }> = []
  const lines = content.split(/\r?\n/)
  const relPath = path.relative(scanRoot, sourcePath) || sourcePath
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#")) continue
    const match = trimmed.match(/^(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.+)$/)
    if (!match) continue
    const varName = match[1]
    if (!KEY_VAR_NAMES.has(varName)) continue
    let value = match[2].trim()
    if (!value || value.startsWith("$")) continue
    if (
      (value.startsWith("\"") && value.endsWith("\"")) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1)
    } else {
      value = value.split(/\s+#/)[0].trim()
    }
    if (!value) continue
    results.push({ key: value, source: relPath, varName })
  }
  return results
}

export async function walkEnvDir(
  dir: string,
  results: Map<string, EnvKeyOption>,
  scanRoot: string = config.envKeyScanRoot,
): Promise<void> {
  let entries: Array<import("node:fs").Dirent>
  try {
    entries = await fs.readdir(dir, { withFileTypes: true })
  } catch {
    return
  }
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      if (IGNORED_DIRS.has(entry.name)) continue
      await walkEnvDir(fullPath, results, scanRoot)
      continue
    }
    if (!entry.isFile()) continue
    if (!/^\.env(\.|$)/.test(entry.name)) continue
    try {
      const stat = await fs.stat(fullPath)
      if (stat.size > 256 * 1024) continue
      const content = await fs.readFile(fullPath, "utf8")
      const found = parseEnvKeys(content, fullPath, scanRoot)
      for (const item of found) {
        const existing = results.get(item.key)
        if (existing) {
          if (!existing.sources.includes(item.source)) {
            existing.sources.push(item.source)
          }
          if (!existing.varNames.includes(item.varName)) {
            existing.varNames.push(item.varName)
          }
        } else {
          results.set(item.key, {
            key: item.key,
            sources: [item.source],
            varNames: [item.varName],
          })
        }
      }
    } catch {
      // ignore parse errors
    }
  }
}

export async function scanEnvKeys(rootDir: string): Promise<EnvKeyOption[]> {
  const results = new Map<string, EnvKeyOption>()
  await walkEnvDir(rootDir, results, rootDir)
  
  // If no keys found, try parent directories up to 3 levels
  if (results.size === 0) {
    let currentDir = rootDir
    for (let i = 0; i < 3; i++) {
      const parentDir = path.dirname(currentDir)
      if (parentDir === currentDir) break // reached root
      currentDir = parentDir
      await walkEnvDir(currentDir, results, currentDir)
      if (results.size > 0) break
    }
  }
  
  return Array.from(results.values())
}

/** Returns the scan root being used for display purposes */
export function getScanRoot(): string {
  return config.envKeyScanRoot
}
