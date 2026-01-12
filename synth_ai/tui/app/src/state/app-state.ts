/**
 * Global application state.
 */

import type { ActivePane, BackendConfig, BackendId, BackendKeySource, FrontendUrlId, LogSource } from "../types"

function normalizeFrontendId(url: string): FrontendUrlId {
  const trimmed = url.trim()
  if (!trimmed) return "unknown"
  try {
    return new URL(trimmed).host || trimmed
  } catch {
    return trimmed.replace(/^https?:\/\//, "").replace(/\/+$/, "")
  }
}

/** Normalize backend ID from env string */
export function normalizeBackendId(value: string): BackendId {
  const lower = value.toLowerCase().trim()
  if (lower === "dev" || lower === "development") return "dev"
  if (lower === "local" || lower === "localhost") return "local"
  return "prod"
}

/** Get frontend URL identifier for a backend (keys are shared by frontend URL) */
export function getFrontendUrlId(_backendId: BackendId): FrontendUrlId {
  return normalizeFrontendId(process.env.SYNTH_FRONTEND_URL || "")
}

/** Get frontend URL for a backend (used for auth and billing pages) */
export function getFrontendUrl(_backendId: BackendId): string {
  return (process.env.SYNTH_FRONTEND_URL || "").trim()
}

type TuiUrlProfile = {
  backendUrl: string
  frontendUrl: string
}

function emptyProfiles(): Record<BackendId, TuiUrlProfile> {
  return {
    prod: { backendUrl: "", frontendUrl: "" },
    dev: { backendUrl: "", frontendUrl: "" },
    local: { backendUrl: "", frontendUrl: "" },
  }
}

function loadTuiProfiles(): Record<BackendId, TuiUrlProfile> {
  const raw = process.env.SYNTH_TUI_URL_PROFILES || ""
  if (!raw) {
    return emptyProfiles()
  }

  try {
    const parsed = JSON.parse(raw) as Record<string, Partial<TuiUrlProfile>>
    const read = (key: string): TuiUrlProfile => ({
      backendUrl: String(parsed?.[key]?.backendUrl || ""),
      frontendUrl: String(parsed?.[key]?.frontendUrl || ""),
    })
    return {
      prod: read("prod"),
      dev: read("dev"),
      local: read("local"),
    }
  } catch {
    return emptyProfiles()
  }
}

const urlProfiles = loadTuiProfiles()

// Backend configurations
export const backendConfigs: Record<BackendId, BackendConfig> = {
  prod: {
    id: "prod",
    label: "Prod",
    backendUrl: urlProfiles.prod.backendUrl,
    frontendUrl: urlProfiles.prod.frontendUrl,
  },
  dev: {
    id: "dev",
    label: "Dev",
    backendUrl: urlProfiles.dev.backendUrl,
    frontendUrl: urlProfiles.dev.frontendUrl,
  },
  local: {
    id: "local",
    label: "Local",
    backendUrl: urlProfiles.local.backendUrl,
    frontendUrl: urlProfiles.local.frontendUrl,
  },
}

// API keys per frontend URL (keys are shared by frontend URL, not backend mode)
const initialFrontendId = normalizeFrontendId(process.env.SYNTH_FRONTEND_URL || "")
export const frontendKeys: Record<FrontendUrlId, string> = initialFrontendId
  ? { [initialFrontendId]: process.env.SYNTH_API_KEY || "" }
  : {}

// Key source tracking (for display purposes)
export const frontendKeySources: Record<FrontendUrlId, BackendKeySource> = {}

/** Get API key for a backend (looks up by frontend URL) */
export function getKeyForBackend(backendId: BackendId): string {
  return frontendKeys[getFrontendUrlId(backendId)]
}

/** Set API key for a backend (stores by frontend URL) */
export function setKeyForBackend(backendId: BackendId, key: string): void {
  frontendKeys[getFrontendUrlId(backendId)] = key
}

/** Get key source for a backend (looks up by frontend URL) */
export function getKeySourceForBackend(backendId: BackendId): BackendKeySource {
  return frontendKeySources[getFrontendUrlId(backendId)]
}

/** Set key source for a backend (stores by frontend URL) */
export function setKeySourceForBackend(backendId: BackendId, source: BackendKeySource): void {
  frontendKeySources[getFrontendUrlId(backendId)] = source
}

// Mutable app state
export const appState = {
  // Backend state
  currentBackend: normalizeBackendId(process.env.SYNTH_TUI_MODE || "prod") as BackendId,

  activePane: "jobs" as ActivePane,
  healthStatus: "unknown",
  autoSelected: false,

  // Event state
  lastSeq: 0,
  selectedEventIndex: 0,
  eventWindowStart: 0,
  eventFilter: "",

  // Job filter state
  jobStatusFilter: new Set<string>(),
  jobFilterOptions: [] as Array<{ status: string; count: number }>,
  jobFilterCursor: 0,
  jobFilterWindowStart: 0,

  // Key modal state
  keyModalBackend: "prod" as BackendId,
  keyPasteActive: false,
  keyPasteBuffer: "",

  // Settings modal state
  settingsCursor: 0,
  settingsOptions: [] as BackendConfig[],

  // Usage modal state
  usageModalOffset: 0,

  // Modal scroll offsets
  eventModalOffset: 0,
  configModalOffset: 0,
  logsModalOffset: 0,
  metricsModalOffset: 0,
  logsModalTail: true,
  promptBrowserIndex: 0,
  promptBrowserOffset: 0,

  // Task Apps modal state
  taskAppsModalOffset: 0,
  taskAppsModalSelectedIndex: 0,

  // Create Job modal state
  createJobCursor: 0,

  // Deploy state
  deployedUrl: null as string | null,
  deployProc: null as import("child_process").ChildProcess | null,

  // Logs pane state
  logsActiveDeploymentId: null as string | null,
  logsSourceFilter: new Set<LogSource>(["uvicorn", "cloudflare", "app"]),
  logsSelectedIndex: 0,
  logsWindowStart: 0,
  logsTailMode: true,

  // Request tokens for cancellation
  jobSelectToken: 0,
  eventsToken: 0,

  // OpenCode state
  principalPane: "jobs" as "jobs" | "opencode",
  openCodeSessionId: null as string | null,
  openCodeUrl: null as string | null,
  openCodeStatus: null as string | null,
  openCodeAutoConnectAttempted: false,
  openCodeMessages: [] as Array<{
    id: string
    role: "user" | "assistant" | "tool"
    content: string
    timestamp: Date
    toolName?: string
    toolStatus?: "pending" | "running" | "completed" | "failed"
  }>,
  openCodeScrollOffset: 0,
  openCodeInputValue: "",
  openCodeIsProcessing: false,

  // Metrics panel view state
  metricsView: "latest" as "latest" | "charts",
}

export function setActivePane(pane: ActivePane): void {
  appState.activePane = pane
}
