/**
 * Scanner for LocalAPI files in a directory.
 * Ported from feat/job-details branch.
 */
import * as fs from "fs"
import * as path from "path"

export interface ScannedLocalAPI {
  filename: string
  filepath: string
}

/**
 * Scan a directory for LocalAPI files.
 * Detection: file contains `from synth_ai.sdk.localapi import` or `create_local_api(`
 */
export function scanForLocalAPIs(directory: string): ScannedLocalAPI[] {
  const results: ScannedLocalAPI[] = []

  try {
    const entries = fs.readdirSync(directory, { withFileTypes: true })

    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(".py")) continue

      const filepath = path.join(directory, entry.name)
      try {
        const content = fs.readFileSync(filepath, "utf-8")
        if (isLocalAPIFile(content)) {
          results.push({
            filename: entry.name,
            filepath,
          })
        }
      } catch {
        // Skip files we can't read
      }
    }
  } catch {
    // Directory doesn't exist or can't be read
  }

  return results
}

/**
 * Scan multiple directories for LocalAPI files.
 */
export function scanMultipleDirectories(directories: string[]): ScannedLocalAPI[] {
  const results: ScannedLocalAPI[] = []
  const seen = new Set<string>()

  for (const dir of directories) {
    for (const api of scanForLocalAPIs(dir)) {
      if (!seen.has(api.filepath)) {
        seen.add(api.filepath)
        results.push(api)
      }
    }
  }

  return results
}

/**
 * Check if file content indicates a LocalAPI file.
 */
function isLocalAPIFile(content: string): boolean {
  return (
    content.includes("from synth_ai.sdk.localapi import") ||
    content.includes("create_local_api(")
  )
}

