/**
 * Persisted settings for the TUI (backend selection + API keys).
 *
 * API keys are stored for the active frontend URL only.
 */
import path from "node:path"
import { promises as fs } from "node:fs"

import { formatEnvLine, parseEnvFile } from "../utils/env"
import { normalizeFrontendId } from "../utils/frontend-id"
import type { BackendId, BackendKeySource, FrontendUrlId } from "../types"

// Type declaration for Node.js process (available at runtime)
declare const process: {
  env: Record<string, string | undefined>
}

export type LoadSettingsDeps = {
  settingsFilePath: string
  normalizeBackendId: (value: string) => BackendId
  setCurrentBackend: (id: BackendId) => void
  setFrontendKey: (id: FrontendUrlId, key: string) => void
  setFrontendKeySource: (id: FrontendUrlId, source: BackendKeySource) => void
}

export async function loadPersistedSettings(deps: LoadSettingsDeps): Promise<void> {
  const {
    settingsFilePath,
    normalizeBackendId,
    setCurrentBackend,
    setFrontendKey,
    setFrontendKeySource,
  } = deps

  try {
    const content = await fs.readFile(settingsFilePath, "utf8")
    const values = parseEnvFile(content)

    const backend = values.SYNTH_TUI_MODE
    if (backend) {
      setCurrentBackend(normalizeBackendId(backend))
    }

    const frontendId = normalizeFrontendId(process.env.SYNTH_FRONTEND_URL || "")

    const apiKey = typeof values.SYNTH_TUI_API_KEY === "string"
      ? values.SYNTH_TUI_API_KEY.trim()
      : ""
    if (apiKey && frontendId) {
      setFrontendKey(frontendId, apiKey)
      setFrontendKeySource(frontendId, {
        sourcePath: values.SYNTH_TUI_API_KEY_SOURCE || null,
        varName: values.SYNTH_TUI_API_KEY_VAR || null,
      })
    }
  } catch (err: any) {
    if (err?.code !== "ENOENT") {
      // Ignore missing file, keep other errors silent for now.
    }
  }
}

export type PersistSettingsDeps = {
  settingsFilePath: string
  getCurrentBackend: () => BackendId
  getFrontendKey: (id: FrontendUrlId) => string
  getFrontendKeySource: (id: FrontendUrlId) => BackendKeySource
  onError?: (message: string) => void
}

export async function persistSettings(deps: PersistSettingsDeps): Promise<void> {
  const {
    settingsFilePath,
    getCurrentBackend,
    getFrontendKey,
    getFrontendKeySource,
    onError,
  } = deps

  try {
    await fs.mkdir(path.dirname(settingsFilePath), { recursive: true })
    const backend = getCurrentBackend()

    const frontendId = normalizeFrontendId(process.env.SYNTH_FRONTEND_URL || "")
    const source = frontendId ? getFrontendKeySource(frontendId) : { sourcePath: null, varName: null }
    const apiKey = frontendId ? getFrontendKey(frontendId) : ""

    const lines = [
      "# synth-ai tui settings",
      formatEnvLine("SYNTH_TUI_MODE", backend),
      formatEnvLine("SYNTH_TUI_API_KEY", apiKey),
      formatEnvLine("SYNTH_TUI_API_KEY_SOURCE", source.sourcePath || ""),
      formatEnvLine("SYNTH_TUI_API_KEY_VAR", source.varName || ""),
    ]
    await fs.writeFile(settingsFilePath, `${lines.join("\n")}\n`, "utf8")
  } catch (err: any) {
    onError?.(`Failed to save settings: ${err?.message || "unknown"}`)
  }
}
