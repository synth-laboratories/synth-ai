/**
 * Global application state.
 */

import type { ActivePane, FocusTarget, LogSource, Mode } from "../types"
import {
  modeUrls,
  modeKeys,
  switchMode as coreSwitchMode,
  getCurrentMode,
  setModeKey,
  setCurrentMode,
  initModeState,
} from "./mode"

// Re-export mode-related items
export { modeUrls, modeKeys, setModeKey, setCurrentMode, initModeState }

/** Switch to a different mode - updates env vars and app state */
export function switchMode(mode: Mode): void {
  coreSwitchMode(mode)
  appState.currentMode = mode
}

// Mutable app state
export const appState = {
  // Mode state (initialized later via initModeState)
  currentMode: getCurrentMode(),

  activePane: "jobs" as ActivePane,
  focusTarget: "list" as FocusTarget,
  healthStatus: "unknown",
  autoSelected: false,

  loginPromptDismissed: false,

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
