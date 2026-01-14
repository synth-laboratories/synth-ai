/**
 * Persisted settings for the TUI (mode selection + API keys per mode).
 */
import path from "node:path"
import { promises as fs } from "node:fs"

import { parseMode } from "../types"
import type { Mode } from "../types"
import { tuiSettingsPath } from "../paths"

export type PersistedSettings = {
  mode: Mode | null
  keys: Record<Mode, string>
}

function getEmptyKeys(): Record<Mode, string> {
  return { prod: "", dev: "", local: "" }
}

export async function readPersistedSettings(): Promise<PersistedSettings> {
  let mode: Mode | null = null
  let keys = getEmptyKeys()
  try {
    const content = await fs.readFile(tuiSettingsPath, "utf8")
    const data = JSON.parse(content) as Partial<PersistedSettings>
    if (data?.mode) {
      mode = parseMode(String(data.mode))
    }
    if (data?.keys && typeof data.keys === "object") {
      const candidate = data.keys as Record<string, unknown>
      keys = {
        prod: typeof candidate.prod === "string" ? candidate.prod.trim() : "",
        dev: typeof candidate.dev === "string" ? candidate.dev.trim() : "",
        local: typeof candidate.local === "string" ? candidate.local.trim() : "",
      }
    }
  } catch (err: any) {
    if (err?.code !== "ENOENT") {
      // Ignore missing file, keep other errors silent for now.
    }
  }
  return { mode, keys }
}

export type LoadSettingsDeps = {
  setCurrentMode: (mode: Mode) => void
}

export async function loadPersistedSettings(
  deps: LoadSettingsDeps,
): Promise<PersistedSettings> {
  const { setCurrentMode } = deps
  const settings = await readPersistedSettings()
  if (settings.mode) {
    setCurrentMode(settings.mode)
  }
  return settings
}

async function writePersistedSettings(
  settings: PersistedSettings,
  onError?: (message: string) => void,
): Promise<void> {
  try {
    await fs.mkdir(path.dirname(tuiSettingsPath), { recursive: true })
    const mode = settings.mode ?? "prod"
    const keys = settings.keys
    const payload = {
      mode,
      keys: {
        prod: keys.prod,
        dev: keys.dev,
        local: keys.local,
      },
    }
    await fs.writeFile(tuiSettingsPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8")
  } catch (err: any) {
    onError?.(`Failed to save settings: ${err?.message || "unknown"}`)
  }
}

export async function persistModeSelection(
  mode: Mode,
  onError?: (message: string) => void,
): Promise<void> {
  const settings = await readPersistedSettings()
  await writePersistedSettings({ ...settings, mode }, onError)
}

export async function persistModeKey(
  mode: Mode,
  key: string,
  onError?: (message: string) => void,
): Promise<void> {
  const settings = await readPersistedSettings()
  const keys = { ...settings.keys, [mode]: key }
  await writePersistedSettings({ ...settings, keys }, onError)
}
