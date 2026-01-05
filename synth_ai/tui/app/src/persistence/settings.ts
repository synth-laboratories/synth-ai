/**
 * Persisted settings for the TUI (backend selection + API keys).
 *
 * Kept separate from app logic so state can be loaded/saved without pulling in UI.
 */
import path from "node:path"
import { promises as fs } from "node:fs"

import { formatEnvLine, parseEnvFile } from "../utils/env"
import type { BackendId, BackendKeySource } from "../types"

export type LoadSettingsDeps = {
  settingsFilePath: string
  normalizeBackendId: (value: string) => BackendId
  setCurrentBackend: (id: BackendId) => void
  setBackendKey: (id: BackendId, key: string) => void
  setBackendKeySource: (id: BackendId, source: BackendKeySource) => void
}

export async function loadPersistedSettings(deps: LoadSettingsDeps): Promise<void> {
  const {
    settingsFilePath,
    normalizeBackendId,
    setCurrentBackend,
    setBackendKey,
    setBackendKeySource,
  } = deps

  try {
    const content = await fs.readFile(settingsFilePath, "utf8")
    const values = parseEnvFile(content)

    const backend = values.SYNTH_TUI_BACKEND
    if (backend) {
      setCurrentBackend(normalizeBackendId(backend))
    }

    const prodKey = values.SYNTH_TUI_API_KEY_PROD
    const devKey = values.SYNTH_TUI_API_KEY_DEV
    const localKey = values.SYNTH_TUI_API_KEY_LOCAL
    if (typeof prodKey === "string") setBackendKey("prod", prodKey)
    if (typeof devKey === "string") setBackendKey("dev", devKey)
    if (typeof localKey === "string") setBackendKey("local", localKey)

    setBackendKeySource("prod", {
      sourcePath: values.SYNTH_TUI_API_KEY_PROD_SOURCE || null,
      varName: values.SYNTH_TUI_API_KEY_PROD_VAR || null,
    })
    setBackendKeySource("dev", {
      sourcePath: values.SYNTH_TUI_API_KEY_DEV_SOURCE || null,
      varName: values.SYNTH_TUI_API_KEY_DEV_VAR || null,
    })
    setBackendKeySource("local", {
      sourcePath: values.SYNTH_TUI_API_KEY_LOCAL_SOURCE || null,
      varName: values.SYNTH_TUI_API_KEY_LOCAL_VAR || null,
    })
  } catch (err: any) {
    if (err?.code !== "ENOENT") {
      // Ignore missing file, keep other errors silent for now.
    }
  }
}

export type PersistSettingsDeps = {
  settingsFilePath: string
  getCurrentBackend: () => BackendId
  getBackendKey: (id: BackendId) => string
  getBackendKeySource: (id: BackendId) => BackendKeySource
  onError?: (message: string) => void
}

export async function persistSettings(deps: PersistSettingsDeps): Promise<void> {
  const {
    settingsFilePath,
    getCurrentBackend,
    getBackendKey,
    getBackendKeySource,
    onError,
  } = deps

  try {
    await fs.mkdir(path.dirname(settingsFilePath), { recursive: true })
    const backend = getCurrentBackend()

    const prodSource = getBackendKeySource("prod")
    const devSource = getBackendKeySource("dev")
    const localSource = getBackendKeySource("local")

    const lines = [
      "# synth-ai tui settings",
      formatEnvLine("SYNTH_TUI_BACKEND", backend),

      formatEnvLine("SYNTH_TUI_API_KEY_PROD", getBackendKey("prod")),
      formatEnvLine("SYNTH_TUI_API_KEY_PROD_SOURCE", prodSource.sourcePath || ""),
      formatEnvLine("SYNTH_TUI_API_KEY_PROD_VAR", prodSource.varName || ""),

      formatEnvLine("SYNTH_TUI_API_KEY_DEV", getBackendKey("dev")),
      formatEnvLine("SYNTH_TUI_API_KEY_DEV_SOURCE", devSource.sourcePath || ""),
      formatEnvLine("SYNTH_TUI_API_KEY_DEV_VAR", devSource.varName || ""),

      formatEnvLine("SYNTH_TUI_API_KEY_LOCAL", getBackendKey("local")),
      formatEnvLine("SYNTH_TUI_API_KEY_LOCAL_SOURCE", localSource.sourcePath || ""),
      formatEnvLine("SYNTH_TUI_API_KEY_LOCAL_VAR", localSource.varName || ""),
    ]
    await fs.writeFile(settingsFilePath, `${lines.join("\n")}\n`, "utf8")
  } catch (err: any) {
    onError?.(`Failed to save settings: ${err?.message || "unknown"}`)
  }
}


