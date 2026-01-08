/**
 * Polling state and timers.
 */

import path from "node:path"

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
  envKeyVisibleCount: parseInt(process.env.SYNTH_TUI_ENV_KEYS_VISIBLE || "8", 10),
  envKeyScanRoot: process.env.SYNTH_TUI_ENV_SCAN_ROOT || process.cwd(),
  settingsFilePath: process.env.SYNTH_TUI_SETTINGS_FILE || path.join(process.env.HOME || process.cwd(), ".synth-ai", "tui-settings"),
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
  // SSE state (jobs list)
  sseConnected: false,
  sseDisconnect: null as (() => void) | null,
  sseReconnectTimer: null as ReturnType<typeof setTimeout> | null,
  sseReconnectDelay: 1000, // Start with 1s, exponential backoff
  lastSseSeq: 0,
  // Eval job SSE state (per-job streaming)
  evalSseConnected: false,
  evalSseDisconnect: null as (() => void) | null,
  evalSseJobId: null as string | null, // Track which job is connected
  evalSseReconnectTimer: null as ReturnType<typeof setTimeout> | null,
  evalSseReconnectDelay: 1000,
  lastEvalSseSeq: 0,
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

export function clearEvalSseTimer(): void {
  if (pollingState.evalSseReconnectTimer) {
    clearTimeout(pollingState.evalSseReconnectTimer)
    pollingState.evalSseReconnectTimer = null
  }
}

export function disconnectEvalSse(): void {
  if (pollingState.evalSseDisconnect) {
    pollingState.evalSseDisconnect()
    pollingState.evalSseDisconnect = null
  }
  pollingState.evalSseConnected = false
  pollingState.evalSseJobId = null
  pollingState.lastEvalSseSeq = 0
  clearEvalSseTimer()
}
