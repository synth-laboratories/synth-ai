/**
 * Persisted settings for the TUI (backend selection + API keys).
 *
 * Keys are stored by frontend URL, not backend mode:
 * - usesynth.ai: prod backend
 * - localhost:3000: dev and local backends (shared key)
 */
import path from "node:path"
import { promises as fs } from "node:fs"

import { formatEnvLine, parseEnvFile } from "../utils/env"
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

    const backend = values.SYNTH_TUI_BACKEND
    if (backend) {
      setCurrentBackend(normalizeBackendId(backend))
    }

    // Load keys by frontend URL with backward compatibility for old key names
    // New format: SYNTH_TUI_API_KEY_USESYNTH, SYNTH_TUI_API_KEY_LOCALHOST
    // Old format: SYNTH_TUI_API_KEY_PROD, SYNTH_TUI_API_KEY_DEV, SYNTH_TUI_API_KEY_LOCAL
    const usesynthKey = values.SYNTH_TUI_API_KEY_USESYNTH || values.SYNTH_TUI_API_KEY_PROD
    const localhostKeyFromFile = values.SYNTH_TUI_API_KEY_LOCALHOST || values.SYNTH_TUI_API_KEY_DEV || values.SYNTH_TUI_API_KEY_LOCAL
    const localhostKey = (typeof localhostKeyFromFile === "string" && localhostKeyFromFile.trim())
      ? localhostKeyFromFile.trim()
      : (process.env.SYNTH_API_KEY || "").trim()

    if (typeof usesynthKey === "string" && usesynthKey.trim()) {
      setFrontendKey("usesynth.ai", usesynthKey.trim())
    }
    if (localhostKey) {
      setFrontendKey("localhost:3000", localhostKey)
    }

    // Load key sources with backward compatibility
    setFrontendKeySource("usesynth.ai", {
      sourcePath: values.SYNTH_TUI_API_KEY_USESYNTH_SOURCE || values.SYNTH_TUI_API_KEY_PROD_SOURCE || null,
      varName: values.SYNTH_TUI_API_KEY_USESYNTH_VAR || values.SYNTH_TUI_API_KEY_PROD_VAR || null,
    })
    setFrontendKeySource("localhost:3000", {
      sourcePath: values.SYNTH_TUI_API_KEY_LOCALHOST_SOURCE || values.SYNTH_TUI_API_KEY_DEV_SOURCE || values.SYNTH_TUI_API_KEY_LOCAL_SOURCE || null,
      varName: values.SYNTH_TUI_API_KEY_LOCALHOST_VAR || values.SYNTH_TUI_API_KEY_DEV_VAR || values.SYNTH_TUI_API_KEY_LOCAL_VAR || null,
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

    const usesynthSource = getFrontendKeySource("usesynth.ai")
    const localhostSource = getFrontendKeySource("localhost:3000")

    const lines = [
      "# synth-ai tui settings",
      "# Keys are stored by frontend URL (usesynth.ai or localhost:3000)",
      formatEnvLine("SYNTH_TUI_BACKEND", backend),

      "# usesynth.ai (prod)",
      formatEnvLine("SYNTH_TUI_API_KEY_USESYNTH", getFrontendKey("usesynth.ai")),
      formatEnvLine("SYNTH_TUI_API_KEY_USESYNTH_SOURCE", usesynthSource.sourcePath || ""),
      formatEnvLine("SYNTH_TUI_API_KEY_USESYNTH_VAR", usesynthSource.varName || ""),

      "# localhost:3000 (dev/local - shared)",
      formatEnvLine("SYNTH_TUI_API_KEY_LOCALHOST", getFrontendKey("localhost:3000")),
      formatEnvLine("SYNTH_TUI_API_KEY_LOCALHOST_SOURCE", localhostSource.sourcePath || ""),
      formatEnvLine("SYNTH_TUI_API_KEY_LOCALHOST_VAR", localhostSource.varName || ""),
    ]
    await fs.writeFile(settingsFilePath, `${lines.join("\n")}\n`, "utf8")
  } catch (err: any) {
    onError?.(`Failed to save settings: ${err?.message || "unknown"}`)
  }
}
