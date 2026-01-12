/**
 * Global application state.
 */

import type { ActivePane, LogSource, Mode, ModeUrls } from "../types"

/** Normalize mode from env string */
export function normalizeMode(value: string): Mode {
  const lower = value.toLowerCase().trim()
  if (lower === "dev" || lower === "development") return "dev"
  if (lower === "local" || lower === "localhost") return "local"
  return "prod"
}

/** Load URL profiles from launcher env var */
function loadModeUrls(): Record<Mode, ModeUrls> {
  const raw = process.env.SYNTH_TUI_URL_PROFILES || ""
  if (!raw) {
    return {
      prod: { backendUrl: "", frontendUrl: "" },
      dev: { backendUrl: "", frontendUrl: "" },
      local: { backendUrl: "", frontendUrl: "" },
    }
  }

  try {
    const parsed = JSON.parse(raw) as Record<string, Partial<ModeUrls>>
    const read = (key: string): ModeUrls => ({
      backendUrl: String(parsed?.[key]?.backendUrl || ""),
      frontendUrl: String(parsed?.[key]?.frontendUrl || ""),
    })
    return { prod: read("prod"), dev: read("dev"), local: read("local") }
  } catch {
    return {
      prod: { backendUrl: "", frontendUrl: "" },
      dev: { backendUrl: "", frontendUrl: "" },
      local: { backendUrl: "", frontendUrl: "" },
    }
  }
}

/** URLs for each mode (loaded from SYNTH_TUI_URL_PROFILES) */
export const modeUrls = loadModeUrls()

/** API keys per mode */
export const modeKeys: Record<Mode, string> = {
  prod: "",
  dev: "",
  local: "",
}

// Initialize current mode's key from env
const initialMode = normalizeMode(process.env.SYNTH_TUI_MODE || "prod")
if (process.env.SYNTH_API_KEY) {
  modeKeys[initialMode] = process.env.SYNTH_API_KEY
}

/** Switch to a different mode - updates env vars and state */
export function switchMode(mode: Mode): void {
  const urls = modeUrls[mode]
  process.env.SYNTH_TUI_MODE = mode
  process.env.SYNTH_BACKEND_URL = urls.backendUrl
  process.env.SYNTH_FRONTEND_URL = urls.frontendUrl
  process.env.SYNTH_API_KEY = modeKeys[mode] || ""
  appState.currentMode = mode
}

// Mutable app state
export const appState = {
  // Mode state
  currentMode: initialMode,

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
  keyModalMode: "prod" as Mode,
  keyPasteActive: false,
  keyPasteBuffer: "",

  // Settings modal state
  settingsCursor: 0,
  settingsOptions: [] as Mode[],

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
