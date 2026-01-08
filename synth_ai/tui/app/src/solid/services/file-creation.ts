/**
 * File creation service for LocalAPI files.
 */
import * as fs from "fs"
import * as path from "path"
import { LOCALAPI_TEMPLATE } from "../templates/localapi"
import { getUniqueFilename, toDisplayPath } from "../utils/files"

export interface CreateFileResult {
  success: boolean
  filePath?: string
  displayPath?: string
  error?: string
}

/**
 * Create a new LocalAPI file in the specified directory.
 * Creates the directory if it doesn't exist.
 * 
 * @param directory - Directory to create the file in
 * @param baseName - Base name for the file (default: "localapi")
 * @returns Result with file path on success, error on failure
 */
export function createLocalApiFile(
  directory: string,
  baseName: string = "localapi",
): CreateFileResult {
  try {
    // Ensure directory exists
    fs.mkdirSync(directory, { recursive: true })
    
    // Get unique filename (adds timestamp if file exists)
    const filePath = getUniqueFilename(directory, baseName, ".py")
    
    // Write the template
    fs.writeFileSync(filePath, LOCALAPI_TEMPLATE, "utf-8")
    
    return {
      success: true,
      filePath,
      displayPath: toDisplayPath(filePath),
    }
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : String(err),
    }
  }
}

/**
 * Open a file in the user's default editor.
 * Uses $EDITOR if set, otherwise falls back to 'open' on macOS.
 */
export function openInEditor(filePath: string): boolean {
  try {
    const { spawn } = require("child_process")
    const editor = process.env.EDITOR
    
    if (editor) {
      spawn(editor, [filePath], {
        detached: true,
        stdio: "ignore",
      }).unref()
    } else {
      // Use 'open' on macOS to open with default app
      spawn("open", [filePath], {
        detached: true,
        stdio: "ignore",
      }).unref()
    }
    
    return true
  } catch {
    return false
  }
}

/**
 * Check if a file exists.
 */
export function fileExists(filePath: string): boolean {
  try {
    fs.accessSync(filePath, fs.constants.F_OK)
    return true
  } catch {
    return false
  }
}

/**
 * Get the filename from a path.
 */
export function getFileName(filePath: string): string {
  return path.basename(filePath)
}

