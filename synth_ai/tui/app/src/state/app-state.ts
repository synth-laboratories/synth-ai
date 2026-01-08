/**
 * Global application state.
 */

import type { ActivePane, BackendConfig, BackendId, BackendKeySource, EnvKeyOption, LogSource } from "../types"

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

// API keys per backend (SYNTH_API_KEY is used as fallback for local and dev)
export const backendKeys: Record<BackendId, string> = {
  prod: process.env.SYNTH_TUI_API_KEY_PROD || process.env.SYNTH_API_KEY || "",
  dev: process.env.SYNTH_TUI_API_KEY_DEV || process.env.SYNTH_API_KEY || "",
  local: process.env.SYNTH_TUI_API_KEY_LOCAL || process.env.SYNTH_API_KEY || "",
}

// Key source tracking (for display purposes)
export const backendKeySources: Record<BackendId, BackendKeySource> = {
  prod: { sourcePath: null, varName: null },
  dev: { sourcePath: null, varName: null },
  local: { sourcePath: null, varName: null },
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
}

export function setActivePane(pane: ActivePane): void {
  appState.activePane = pane
}
