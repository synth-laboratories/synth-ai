/**
 * Polling state and timers.
 */

import { registerCleanup } from "../lifecycle"
import { tuiSettingsPath } from "../paths"

// Configuration from environment
export const config = {
  initialJobId: process.env.SYNTH_TUI_JOB_ID || "",
  refreshInterval: parseFloat(process.env.SYNTH_TUI_REFRESH_INTERVAL || "5"),
  eventInterval: parseFloat(process.env.SYNTH_TUI_EVENT_INTERVAL || "2"),
  maxRefreshInterval: parseFloat(process.env.SYNTH_TUI_REFRESH_MAX || "60"),
  maxEventInterval: parseFloat(process.env.SYNTH_TUI_EVENT_MAX || "15"),
  eventHistoryLimit: parseInt(process.env.SYNTH_TUI_EVENT_CARDS || "200", 10),
  eventCollapseLimit: parseInt(process.env.SYNTH_TUI_EVENT_COLLAPSE || "160", 10),
  eventVisibleCount: parseInt(process.env.SYNTH_TUI_EVENT_VISIBLE || "6", 10),
  jobLimit: parseInt(process.env.SYNTH_TUI_LIMIT || "50", 10),
  settingsFilePath: tuiSettingsPath,
  jobFilterVisibleCount: 6,
}

// Polling state
export const pollingState = {
  jobsPollMs: Math.max(1, config.refreshInterval) * 1000,
  eventsPollMs: Math.max(0.5, config.eventInterval) * 1000,
  jobsInFlight: false,
  eventsInFlight: false,
  jobsTimer: null as ReturnType<typeof setTimeout> | null,
  eventsTimer: null as ReturnType<typeof setTimeout> | null,
  // SSE state
  sseConnected: false,
  sseDisconnect: null as (() => void) | null,
  sseReconnectTimer: null as ReturnType<typeof setTimeout> | null,
  sseReconnectDelay: 1000, // Start with 1s, exponential backoff
  lastSseSeq: 0,
}

export function clearJobsTimer(): void {
  if (pollingState.jobsTimer) {
    clearTimeout(pollingState.jobsTimer)
    pollingState.jobsTimer = null
  }
}

export function clearEventsTimer(): void {
  if (pollingState.eventsTimer) {
    clearTimeout(pollingState.eventsTimer)
    pollingState.eventsTimer = null
  }
}

function clearSseReconnectTimer(): void {
  if (pollingState.sseReconnectTimer) {
    clearTimeout(pollingState.sseReconnectTimer)
    pollingState.sseReconnectTimer = null
  }
}

// Register cleanup handlers for all polling timers
registerCleanup("polling-jobs-timer", clearJobsTimer)
registerCleanup("polling-events-timer", clearEventsTimer)
registerCleanup("polling-sse-reconnect", clearSseReconnectTimer)
registerCleanup("polling-sse-disconnect", () => {
  if (pollingState.sseDisconnect) {
    pollingState.sseDisconnect()
    pollingState.sseDisconnect = null
  }
})
