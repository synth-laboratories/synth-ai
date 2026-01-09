/**
 * Global application state.
 */

import type { ActivePane, BackendConfig, BackendId, BackendKeySource, EnvKeyOption, FrontendUrlId, LogSource } from "../types"

/** Ensure URL ends with /api */
function ensureApiBase(url: string): string {
  let base = url.trim().replace(/\/+$/, "")
  if (!base.endsWith("/api")) {
    base = base + "/api"
  }
  return base
}

/** Normalize backend ID from env string */
export function normalizeBackendId(value: string): BackendId {
  const lower = value.toLowerCase().trim()
  if (lower === "dev" || lower === "development") return "dev"
  if (lower === "local" || lower === "localhost") return "local"
  return "prod"
}

/** Get frontend URL identifier for a backend (keys are shared by frontend URL) */
export function getFrontendUrlId(backendId: BackendId): FrontendUrlId {
  switch (backendId) {
    case "prod": return "usesynth.ai"
    case "dev":
    case "local": return "localhost:3000"
  }
}

/** Get frontend URL for a backend (used for auth and billing pages) */
export function getFrontendUrl(backendId: BackendId): string {
  switch (backendId) {
    case "prod": return "https://usesynth.ai"
    case "dev":
    case "local": return "http://localhost:3000"
  }
}

// Backend configurations
export const backendConfigs: Record<BackendId, BackendConfig> = {
  prod: {
    id: "prod",
    label: "Prod",
    baseUrl: ensureApiBase(
      process.env.SYNTH_TUI_PROD_API_BASE || "https://api.usesynth.ai/api",
    ),
  },
  dev: {
    id: "dev",
    label: "Dev",
    baseUrl: ensureApiBase(
      process.env.SYNTH_TUI_DEV_API_BASE || "https://synth-backend-dev-docker.onrender.com/api",
    ),
  },
  local: {
    id: "local",
    label: "Local",
    baseUrl: ensureApiBase(
      process.env.SYNTH_TUI_LOCAL_API_BASE || "http://localhost:8000/api",
    ),
  },
}

// API keys per frontend URL (keys are shared by frontend URL, not backend mode)
// dev and local both use localhost:3000, so they share the same key
export const frontendKeys: Record<FrontendUrlId, string> = {
  "usesynth.ai": process.env.SYNTH_TUI_API_KEY_PROD || process.env.SYNTH_API_KEY || "",
  "localhost:3000": process.env.SYNTH_TUI_API_KEY_LOCAL || process.env.SYNTH_API_KEY || "",
}

// Key source tracking (for display purposes)
export const frontendKeySources: Record<FrontendUrlId, BackendKeySource> = {
  "usesynth.ai": { sourcePath: null, varName: null },
  "localhost:3000": { sourcePath: null, varName: null },
}

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
  currentBackend: normalizeBackendId(process.env.SYNTH_TUI_BACKEND || "prod") as BackendId,

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

  // Env key modal state
  envKeyOptions: [] as EnvKeyOption[],
  envKeyCursor: 0,
  envKeyWindowStart: 0,
  envKeyScanInProgress: false,
  envKeyError: null as string | null,

  // Usage modal state
  usageModalOffset: 0,

  // Modal scroll offsets
  eventModalOffset: 0,
  resultsModalOffset: 0,
  configModalOffset: 0,
  logsModalOffset: 0,
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
}

export function setActivePane(pane: ActivePane): void {
  appState.activePane = pane
}
