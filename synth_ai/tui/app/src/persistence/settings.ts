/**
 * Persisted settings for the TUI (mode selection, API keys, list filters).
 */
import path from "node:path"
import { promises as fs } from "node:fs"

import { ListPane, parseMode } from "../types"
import type { ListFilterMode, Mode } from "../types"
import { tuiSettingsPath } from "../paths"

export type PersistedListFilter = {
  mode: ListFilterMode
  selections: string[]
}

export type PersistedListFilters = Record<Mode, Record<ListPane, PersistedListFilter>>

export type PersistedSettings = {
  mode: Mode | null
  keys: Record<Mode, string>
  listFilters: PersistedListFilters
}

function getEmptyKeys(): Record<Mode, string> {
  return { prod: "", dev: "", local: "" }
}

function createDefaultListFilter(): PersistedListFilter {
  return { mode: "all", selections: [] }
}

function getDefaultListFilters(): PersistedListFilters {
  return {
    prod: {
      [ListPane.Jobs]: createDefaultListFilter(),
      [ListPane.Logs]: createDefaultListFilter(),
    },
    dev: {
      [ListPane.Jobs]: createDefaultListFilter(),
      [ListPane.Logs]: createDefaultListFilter(),
    },
    local: {
      [ListPane.Jobs]: createDefaultListFilter(),
      [ListPane.Logs]: createDefaultListFilter(),
    },
  }
}

function normalizeListFilterMode(value: unknown): ListFilterMode {
  if (value === "all" || value === "none" || value === "subset") {
    return value
  }
  return "all"
}

function normalizeListFilterSelections(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const seen = new Set<string>()
  for (const entry of value) {
    if (typeof entry !== "string") continue
    const trimmed = entry.trim()
    if (!trimmed || seen.has(trimmed)) continue
    seen.add(trimmed)
  }
  return Array.from(seen)
}

function normalizePersistedListFilter(value: unknown): PersistedListFilter {
  if (!value || typeof value !== "object") return createDefaultListFilter()
  const candidate = value as { mode?: unknown; selections?: unknown }
  const selections = normalizeListFilterSelections(candidate.selections)
  let mode = normalizeListFilterMode(candidate.mode)
  if (mode === "subset" && selections.length === 0) {
    mode = "none"
  }
  if (mode === "all" || mode === "none") {
    return { mode, selections: [] }
  }
  return { mode, selections }
}

function normalizePersistedListFilters(value: unknown): PersistedListFilters {
  const defaults = getDefaultListFilters()
  if (!value || typeof value !== "object") return defaults
  const candidate = value as Record<string, unknown>
  const normalizePane = (modeKey: Mode, pane: ListPane): PersistedListFilter => {
    const modeEntry = candidate[modeKey]
    if (!modeEntry || typeof modeEntry !== "object") return defaults[modeKey][pane]
    const paneEntry = (modeEntry as Record<string, unknown>)[pane]
    return normalizePersistedListFilter(paneEntry)
  }
  return {
    prod: {
      [ListPane.Jobs]: normalizePane("prod", ListPane.Jobs),
      [ListPane.Logs]: normalizePane("prod", ListPane.Logs),
    },
    dev: {
      [ListPane.Jobs]: normalizePane("dev", ListPane.Jobs),
      [ListPane.Logs]: normalizePane("dev", ListPane.Logs),
    },
    local: {
      [ListPane.Jobs]: normalizePane("local", ListPane.Jobs),
      [ListPane.Logs]: normalizePane("local", ListPane.Logs),
    },
  }
}

export async function readPersistedSettings(): Promise<PersistedSettings> {
  let mode: Mode | null = null
  let keys = getEmptyKeys()
  let listFilters = getDefaultListFilters()
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
    if (data?.listFilters) {
      listFilters = normalizePersistedListFilters(data.listFilters)
    }
  } catch (err: any) {
    if (err?.code !== "ENOENT") {
      // Ignore missing file, keep other errors silent for now.
    }
  }
  return { mode, keys, listFilters }
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

function sanitizeListFilters(filters: Record<ListPane, PersistedListFilter>): Record<ListPane, PersistedListFilter> {
  return {
    [ListPane.Jobs]: normalizePersistedListFilter(filters[ListPane.Jobs]),
    [ListPane.Logs]: normalizePersistedListFilter(filters[ListPane.Logs]),
  }
}

async function writePersistedSettings(
  settings: PersistedSettings,
  onError?: (message: string) => void,
): Promise<void> {
  try {
    await fs.mkdir(path.dirname(tuiSettingsPath), { recursive: true })
    const mode = settings.mode ?? "prod"
    const keys = settings.keys
    const listFilters = settings.listFilters ?? getDefaultListFilters()
    const payload = {
      mode,
      keys: {
        prod: keys.prod,
        dev: keys.dev,
        local: keys.local,
      },
      listFilters,
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

export async function persistListFilters(
  mode: Mode,
  filters: Record<ListPane, PersistedListFilter>,
  onError?: (message: string) => void,
): Promise<void> {
  const settings = await readPersistedSettings()
  const listFilters = {
    ...settings.listFilters,
    [mode]: sanitizeListFilters(filters),
  }
  await writePersistedSettings({ ...settings, listFilters }, onError)
}
