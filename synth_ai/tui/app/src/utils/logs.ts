import fs from "node:fs"
import path from "node:path"
import { tuiLogsDir } from "../paths"

export type LogFileInfo = {
  path: string
  name: string
  mtimeMs: number
  birthtimeMs: number
  size: number
}

export type LogChunk = {
  text: string
  size: number
  truncated: boolean
}

export function getLogsDirectory(): string {
  return tuiLogsDir
}

export async function listLogFiles(): Promise<LogFileInfo[]> {
  const logsDir = getLogsDirectory()
  try {
    const entries = await fs.promises.readdir(logsDir, { withFileTypes: true })
    const files = await Promise.all(
      entries
        .filter((entry) => entry.isFile())
        .map(async (entry) => {
          const fullPath = path.join(logsDir, entry.name)
          try {
            const stat = await fs.promises.stat(fullPath)
            if (!stat.isFile()) return null
            return {
              path: fullPath,
              name: entry.name,
              mtimeMs: stat.mtimeMs,
              birthtimeMs: stat.birthtimeMs,
              size: stat.size,
            } satisfies LogFileInfo
          } catch {
            return null
          }
        }),
    )
    return files
      .filter((file): file is LogFileInfo => Boolean(file))
      .sort((a, b) => {
        // Filenames are {timestamp}_{type}.log
        // Sort by timestamp descending, then type ascending (A-Z)
        const aBase = a.name.replace(/\.log$/, "")
        const bBase = b.name.replace(/\.log$/, "")
        const aLastUnderscore = aBase.lastIndexOf("_")
        const bLastUnderscore = bBase.lastIndexOf("_")

        const aTimestamp = aLastUnderscore > 0 ? aBase.slice(0, aLastUnderscore) : aBase
        const bTimestamp = bLastUnderscore > 0 ? bBase.slice(0, bLastUnderscore) : bBase
        const aType = aLastUnderscore > 0 ? aBase.slice(aLastUnderscore + 1) : ""
        const bType = bLastUnderscore > 0 ? bBase.slice(bLastUnderscore + 1) : ""

        // Timestamp descending (newer first)
        const timestampCmp = bTimestamp.localeCompare(aTimestamp)
        if (timestampCmp !== 0) return timestampCmp

        // Type ascending (A-Z)
        return aType.localeCompare(bType)
      })
  } catch {
    return []
  }
}

export async function readLogFile(filePath: string): Promise<string> {
  try {
    return await fs.promises.readFile(filePath, "utf8")
  } catch (err: any) {
    return `Failed to read ${filePath}: ${err?.message || String(err)}`
  }
}

export async function readLogChunk(filePath: string, from: number): Promise<LogChunk | null> {
  let handle: fs.promises.FileHandle | null = null
  try {
    handle = await fs.promises.open(filePath, "r")
    const stat = await handle.stat()
    const size = stat.size
    if (size <= from) {
      return { text: "", size, truncated: size < from }
    }
    const length = size - from
    const buffer = Buffer.alloc(length)
    await handle.read(buffer, 0, length, from)
    return { text: buffer.toString("utf8"), size, truncated: false }
  } catch {
    return null
  } finally {
    if (handle) {
      await handle.close().catch(() => undefined)
    }
  }
}
