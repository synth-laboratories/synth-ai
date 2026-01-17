/**
 * Polling state and timers.
 */

import { registerCleanup } from "../lifecycle"
// Configuration from environment
export const config = {
  initialJobId: process.env.SYNTH_TUI_JOB_ID || "",
  refreshInterval: parseFloat(process.env.SYNTH_TUI_REFRESH_INTERVAL || "15"),
  eventInterval: parseFloat(process.env.SYNTH_TUI_EVENT_INTERVAL || "2"),
  maxRefreshInterval: parseFloat(process.env.SYNTH_TUI_REFRESH_MAX || "60"),
  maxEventInterval: parseFloat(process.env.SYNTH_TUI_EVENT_MAX || "15"),
  eventHistoryLimit: parseInt(process.env.SYNTH_TUI_EVENT_CARDS || "200", 10),
  eventCollapseLimit: parseInt(process.env.SYNTH_TUI_EVENT_COLLAPSE || "160", 10),
  eventVisibleCount: parseInt(process.env.SYNTH_TUI_EVENT_VISIBLE || "6", 10),
  jobLimit: 25,
  listFilterVisibleCount: 6,
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
  sseDisconnect: null as (() => void) | null,
  sseReconnectTimer: null as ReturnType<typeof setTimeout> | null,
  sseReconnectDelay: 1000, // Start with 1s, exponential backoff
  lastSseSeq: 0,
}

export type SseChannel = "jobs" | "job-events"

const sseChannelState: Record<SseChannel, boolean> = {
  jobs: false,
  "job-events": false,
}

type SseListener = (connected: boolean) => void

const sseListeners: Record<SseChannel, Set<SseListener>> = {
  jobs: new Set(),
  "job-events": new Set(),
}

const pollNextAt: Record<SseChannel, number> = {
  jobs: 0,
  "job-events": 0,
}

export function setSseConnected(channel: SseChannel, connected: boolean): void {
  if (sseChannelState[channel] === connected) {
    return
  }
  sseChannelState[channel] = connected
  for (const listener of sseListeners[channel]) {
    listener(connected)
  }
}

export function isSseConnected(channel: SseChannel): boolean {
  return sseChannelState[channel]
}

export function onSseChange(channel: SseChannel, listener: SseListener): () => void {
  sseListeners[channel].add(listener)
  return () => {
    sseListeners[channel].delete(listener)
  }
}

export function shouldPoll(channel: SseChannel): boolean {
  return !isSseConnected(channel)
}

export function resetSseConnections(): void {
  for (const key of Object.keys(sseChannelState) as SseChannel[]) {
    setSseConnected(key, false)
  }
}

export function setPollNextAt(channel: SseChannel, nextAt: number): void {
  pollNextAt[channel] = nextAt
}

export function clearPollNextAt(channel: SseChannel): void {
  pollNextAt[channel] = 0
}

export function isPollScheduledSoon(channel: SseChannel, withinMs: number): boolean {
  const nextAt = pollNextAt[channel]
  if (!nextAt) return false
  return nextAt - Date.now() <= withinMs
}

export function clearJobsTimer(): void {
  if (pollingState.jobsTimer) {
    clearTimeout(pollingState.jobsTimer)
    pollingState.jobsTimer = null
  }
  clearPollNextAt("jobs")
}

export function clearEventsTimer(): void {
  if (pollingState.eventsTimer) {
    clearTimeout(pollingState.eventsTimer)
    pollingState.eventsTimer = null
  }
  clearPollNextAt("job-events")
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
