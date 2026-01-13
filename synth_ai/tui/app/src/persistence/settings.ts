/**
 * Persisted settings for the TUI (mode selection + API keys per mode).
 */
import path from "node:path"
import { promises as fs } from "node:fs"

import { formatEnvLine, parseEnvFile } from "../utils/env"
import { parseMode } from "../types"
import type { Mode } from "../types"

export type LoadSettingsDeps = {
  settingsFilePath: string
  setCurrentMode: (mode: Mode) => void
  setModeKey: (mode: Mode, key: string) => void
}

export async function loadPersistedSettings(deps: LoadSettingsDeps): Promise<void> {
  const { settingsFilePath, setCurrentMode, setModeKey } = deps

  try {
    const content = await fs.readFile(settingsFilePath, "utf8")
    const values = parseEnvFile(content)

    const mode = values.SYNTH_TUI_MODE
    if (mode) {
      setCurrentMode(parseMode(mode))
    }

    // Load keys per mode
    if (values.SYNTH_TUI_KEY_PROD) setModeKey("prod", values.SYNTH_TUI_KEY_PROD.trim())
    if (values.SYNTH_TUI_KEY_DEV) setModeKey("dev", values.SYNTH_TUI_KEY_DEV.trim())
    if (values.SYNTH_TUI_KEY_LOCAL) setModeKey("local", values.SYNTH_TUI_KEY_LOCAL.trim())
  } catch (err: any) {
    if (err?.code !== "ENOENT") {
      // Ignore missing file, keep other errors silent for now.
    }
  }
}

export type PersistSettingsDeps = {
  settingsFilePath: string
  getCurrentMode: () => Mode
  getModeKeys: () => Record<Mode, string>
  onError?: (message: string) => void
}

export async function persistSettings(deps: PersistSettingsDeps): Promise<void> {
  const { settingsFilePath, getCurrentMode, getModeKeys, onError } = deps

  try {
    await fs.mkdir(path.dirname(settingsFilePath), { recursive: true })
    const mode = getCurrentMode()
    const keys = getModeKeys()

    const lines = [
      "# synth-ai tui settings",
      formatEnvLine("SYNTH_TUI_MODE", mode),
      formatEnvLine("SYNTH_TUI_KEY_PROD", keys.prod),
      formatEnvLine("SYNTH_TUI_KEY_DEV", keys.dev),
      formatEnvLine("SYNTH_TUI_KEY_LOCAL", keys.local),
    ]
    await fs.writeFile(settingsFilePath, `${lines.join("\n")}\n`, "utf8")
  } catch (err: any) {
    onError?.(`Failed to save settings: ${err?.message || "unknown"}`)
  }
}
