/**
 * File path utilities.
 * Ported from feat/job-details branch.
 */
import * as fs from "fs"
import * as path from "path"
import * as os from "os"

/**
 * Convert an absolute path to use ~/ prefix for display.
 * Example: /Users/foo/bar -> ~/bar
 */
export function toDisplayPath(absolutePath: string): string {
  const home = os.homedir()
  if (absolutePath.startsWith(home)) {
    return "~" + absolutePath.slice(home.length)
  }
  return absolutePath
}

/**
 * Expand ~/ prefix to actual home directory.
 * Example: ~/bar -> /Users/foo/bar
 */
export function expandPath(displayPath: string): string {
  if (displayPath.startsWith("~/")) {
    return path.join(os.homedir(), displayPath.slice(2))
  }
  if (displayPath === "~") {
    return os.homedir()
  }
  return displayPath
}

/**
 * Generate a unique filename, adding timestamp suffix if file already exists.
 * Format: {baseName}_{year}_{month}_{day}_{time}{ext}
 * Example: localapi_2024_01_15_143022.py
 */
export function getUniqueFilename(dir: string, baseName: string, ext: string): string {
  const baseFilePath = path.join(dir, `${baseName}${ext}`)
  if (!fs.existsSync(baseFilePath)) {
    return baseFilePath
  }

  // File exists, add timestamp suffix
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, "0")
  const day = String(now.getDate()).padStart(2, "0")
  const hours = String(now.getHours()).padStart(2, "0")
  const minutes = String(now.getMinutes()).padStart(2, "0")
  const seconds = String(now.getSeconds()).padStart(2, "0")
  const time = `${hours}${minutes}${seconds}`

  const newName = `${baseName}_${year}_${month}_${day}_${time}${ext}`
  return path.join(dir, newName)
}

/**
 * Format a timestamp for filenames: YYYY_MM_DD_HH-MM-SS
 */
export function formatTimestampForFilename(date: Date = new Date()): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  const hours = String(date.getHours()).padStart(2, "0")
  const minutes = String(date.getMinutes()).padStart(2, "0")
  const seconds = String(date.getSeconds()).padStart(2, "0")
  return `${year}_${month}_${day}_${hours}-${minutes}-${seconds}`
}

