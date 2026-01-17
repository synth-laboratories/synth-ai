import * as fs from "fs"
import * as path from "path"

export interface ScannedLocalAPI {
  filename: string
  filepath: string
}

export type LocalApiScanCache = Map<string, { mtimeMs: number; isLocalApi: boolean }>

/**
 * Scan a directory for LocalAPI files.
 * Detection: file contains `from synth_ai.sdk.localapi import` or `create_local_api(`
 */
export async function scanForLocalAPIs(
  directory: string,
  cache: LocalApiScanCache,
): Promise<ScannedLocalAPI[]> {
  const results: ScannedLocalAPI[] = []
  const seen = new Set<string>()

  let entries: fs.Dirent[] = []
  try {
    entries = await fs.promises.readdir(directory, { withFileTypes: true })
  } catch {
    return results
  }

  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".py")) continue

    const filepath = path.join(directory, entry.name)
    seen.add(filepath)

    let stat: fs.Stats | null = null
    try {
      stat = await fs.promises.stat(filepath)
    } catch {
      continue
    }

    const cached = cache.get(filepath)
    if (cached && cached.mtimeMs === stat.mtimeMs) {
      if (cached.isLocalApi) {
        results.push({ filename: entry.name, filepath })
      }
      continue
    }

    let content = ""
    try {
      content = await fs.promises.readFile(filepath, "utf-8")
    } catch {
      cache.set(filepath, { mtimeMs: stat.mtimeMs, isLocalApi: false })
      continue
    }

    const isLocal = isLocalAPIFile(content)
    cache.set(filepath, { mtimeMs: stat.mtimeMs, isLocalApi: isLocal })
    if (isLocal) {
      results.push({ filename: entry.name, filepath })
    }
  }

  for (const key of cache.keys()) {
    if (path.dirname(key) === directory && !seen.has(key)) {
      cache.delete(key)
    }
  }

  return results
}

/**
 * Scan multiple directories for LocalAPI files.
 */
export async function scanMultipleDirectories(
  directories: string[],
  cache: LocalApiScanCache,
): Promise<ScannedLocalAPI[]> {
  const results = await Promise.all(
    directories.map((dir) => scanForLocalAPIs(dir, cache)),
  )
  const seen = new Set<string>()
  const combined: ScannedLocalAPI[] = []
  for (const list of results) {
    for (const api of list) {
      if (!seen.has(api.filepath)) {
        seen.add(api.filepath)
        combined.push(api)
      }
    }
  }
  return combined
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
