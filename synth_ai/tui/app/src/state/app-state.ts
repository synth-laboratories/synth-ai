/**
 * Global application state.
 */

import type { ActivePane, FocusTarget, LogSource, Mode } from "../types"
import { ListPane } from "../types"
import { config } from "./polling"
import {
  modeUrls,
  switchMode as coreSwitchMode,
  getCurrentMode,
  setCurrentMode,
  initModeState,
} from "./mode"

// Re-export mode-related items
export { modeUrls, setCurrentMode, initModeState }

/** Switch to a different mode - updates env vars and app state */
export function switchMode(mode: Mode): void {
  coreSwitchMode(mode)
}

export type AppState = {
  currentMode: Mode
  activePane: ActivePane
  focusTarget: FocusTarget
  healthStatus: string
  autoSelected: boolean
  loginPromptDismissed: boolean
  lastSeq: number
  selectedEventId: string | null
  selectedEventIndex: number
  eventWindowStart: number
  eventVisibleCount: number
  verifierEvolveGenerationIndex: number
  eventFilter: string
  settingsCursor: number
  settingsOptions: Mode[]
  settingsKeys: Record<Mode, string>
  listFilterPane: ActivePane
  listFilterOptions: Array<{ id: string; label: string; count: number }>
  listFilterCursor: number
  listFilterWindowStart: number
  listFilterVisibleCount: number
  listFilterSelections: Record<ActivePane, Set<string>>
  jobsListLimit: number
  jobsListLoadingMore: boolean
  jobsListHasMore: boolean
  jobsListServerCount: number
  usageModalOffset: number
  eventModalOffset: number
  configModalOffset: number
  logsModalOffset: number
  metricsModalOffset: number
  logsModalTail: boolean
  promptBrowserIndex: number
  promptBrowserOffset: number
  taskAppsModalOffset: number
  taskAppsModalSelectedIndex: number
  candidatesGenerationFilter: number | null
  createJobCursor: number
  deployedUrl: string | null
  deployProc: import("child_process").ChildProcess | null
  logsActiveDeploymentId: string | null
  logsSourceFilter: Set<LogSource>
  logsSelectedIndex: number
  logsWindowStart: number
  logsTailMode: boolean
  logsDetailOffset: number
  logsDetailTail: boolean
  jobSelectToken: number
  eventsToken: number
  principalPane: "jobs" | "opencode"
  openCodeSessionId: string | null
  openCodeUrl: string | null
  openCodeStatus: string | null
  openCodeAutoConnectAttempted: boolean
  openCodeMessages: Array<{
    id: string
    role: "user" | "assistant" | "tool"
    content: string
    timestamp: Date
    toolName?: string
    toolStatus?: "pending" | "running" | "completed" | "failed"
  }>
  openCodeScrollOffset: number
  openCodeInputValue: string
  openCodeIsProcessing: boolean
  metricsView: "latest" | "charts"
}

export function createInitialAppState(): AppState {
  return {
    // Mode state (initialized later via initModeState)
    currentMode: getCurrentMode(),

    activePane: ListPane.Jobs as ActivePane,
    focusTarget: "list" as FocusTarget,
    healthStatus: "unknown",
    autoSelected: false,

    loginPromptDismissed: false,

    // Event state
    lastSeq: 0,
    selectedEventId: null,
    selectedEventIndex: 0,
    eventWindowStart: 0,
    eventVisibleCount: config.eventVisibleCount,
    verifierEvolveGenerationIndex: 0,
    eventFilter: "",

    // Settings modal state
    settingsCursor: 0,
    settingsOptions: [] as Mode[],
    settingsKeys: { prod: "", dev: "", local: "" } as Record<Mode, string>,

    // List filter state
    listFilterPane: ListPane.Jobs as ActivePane,
    listFilterOptions: [] as Array<{ id: string; label: string; count: number }>,
    listFilterCursor: 0,
    listFilterWindowStart: 0,
    listFilterVisibleCount: config.listFilterVisibleCount,
    listFilterSelections: {
      [ListPane.Jobs]: new Set<string>(),
      [ListPane.Logs]: new Set<string>(),
    } as Record<ActivePane, Set<string>>,
    jobsListLimit: config.jobLimit,
    jobsListLoadingMore: false,
    jobsListHasMore: false,
    jobsListServerCount: 0,

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
    candidatesGenerationFilter: null,

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
    logsDetailOffset: 0,
    logsDetailTail: true,

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
}

export const initialAppState = createInitialAppState()
