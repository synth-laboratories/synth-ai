/**
 * synth.md file utilities
 *
 * Handles discovery, reading, watching, and saving of user's synth.md instructions.
 */

import { watch, type FSWatcher } from "fs"
import { readFile, writeFile, access } from "fs/promises"
import { join } from "path"

export interface SynthMdState {
  path: string | null
  content: string
  lastModified: Date | null
  isEditing: boolean
}

/**
 * Find synth.md file in the working directory.
 * Searches in order:
 * 1. {cwd}/synth.md
 * 2. {cwd}/.synth/synth.md
 */
export async function findSynthMd(cwd: string): Promise<string | null> {
  const candidates = [
    join(cwd, "synth.md"),
    join(cwd, ".synth", "synth.md"),
  ]

  for (const path of candidates) {
    try {
      await access(path)
      return path
    } catch {
      // File doesn't exist, try next
    }
  }

  return null
}

/**
 * Load synth.md content from a file path.
 */
export async function loadSynthMd(path: string): Promise<string> {
  try {
    return await readFile(path, "utf-8")
  } catch {
    return ""
  }
}

/**
 * Save content to synth.md file.
 */
export async function saveSynthMd(path: string, content: string): Promise<boolean> {
  try {
    await writeFile(path, content, "utf-8")
    return true
  } catch {
    return false
  }
}

/**
 * Watch synth.md for changes.
 * Returns a cleanup function to stop watching.
 */
export function watchSynthMd(
  path: string,
  onChange: (content: string) => void
): () => void {
  let debounceTimer: ReturnType<typeof setTimeout> | null = null
  let watcher: FSWatcher | null = null

  try {
    watcher = watch(path, async (eventType) => {
      if (eventType === "change") {
        // Debounce to avoid multiple rapid updates
        if (debounceTimer) clearTimeout(debounceTimer)
        debounceTimer = setTimeout(async () => {
          const content = await loadSynthMd(path)
          onChange(content)
        }, 100)
      }
    })
  } catch {
    // Watch failed, ignore
  }

  return () => {
    if (watcher) {
      watcher.close()
    }
    if (debounceTimer) {
      clearTimeout(debounceTimer)
    }
  }
}

/**
 * Initialize synth.md state for a working directory.
 * Finds the file, loads content, and sets up watching.
 */
export async function initSynthMd(
  cwd: string,
  onChange: (state: SynthMdState) => void
): Promise<{ state: SynthMdState; cleanup: () => void }> {
  const path = await findSynthMd(cwd)

  const initialState: SynthMdState = {
    path,
    content: path ? await loadSynthMd(path) : "",
    lastModified: path ? new Date() : null,
    isEditing: false,
  }

  let cleanup = () => {}

  if (path) {
    cleanup = watchSynthMd(path, (newContent) => {
      onChange({
        ...initialState,
        content: newContent,
        lastModified: new Date(),
      })
    })
  }

  return { state: initialState, cleanup }
}
