/**
 * Event polling operations.
 */
import type { AppContext } from "../context"
import { extractEvents, isEvalJob, type JobEvent } from "../tui_data"
import { apiGet } from "./client"
import { pollingState } from "../state/polling"

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function eventMatchesFilter(event: JobEvent, filter: string): boolean {
  const haystack = [
    event.type,
    event.message,
    event.timestamp,
    event.data ? safeEventDataText(event.data) : "",
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
  return haystack.includes(filter)
}

function safeEventDataText(data: unknown): string {
  if (data == null) return ""
  if (typeof data === "string") return data
  if (typeof data === "number" || typeof data === "boolean") return String(data)
  try {
    return JSON.stringify(data)
  } catch {
    return ""
  }
}

export async function refreshEvents(ctx: AppContext): Promise<boolean> {
  const { snapshot, appState, config } = ctx.state
  const job = snapshot.selectedJob
  if (!job) return true

  // Skip polling if eval SSE is connected for this job
  if (
    isEvalJob(job) &&
    pollingState.evalSseConnected &&
    pollingState.evalSseJobId === job.job_id
  ) {
    return true // SSE handles events for this eval job
  }

  const jobId = job.job_id
  const token = appState.eventsToken

  try {
    const isGepa = job.training_type === "gepa" || job.training_type === "graph_gepa"
    const paths =
      isEvalJob(job)
        ? [
            `/eval/jobs/${job.job_id}/events?since_seq=${appState.lastSeq}&limit=200`,
            `/learning/jobs/${job.job_id}/events?since_seq=${appState.lastSeq}&limit=200`,
          ]
        : job.job_source === "learning"
          ? [`/learning/jobs/${job.job_id}/events?since_seq=${appState.lastSeq}&limit=200`]
          : isGepa
            ? [
                `/prompt-learning/online/jobs/${job.job_id}/events?since_seq=${appState.lastSeq}&limit=200`,
                `/learning/jobs/${job.job_id}/events?since_seq=${appState.lastSeq}&limit=200`,
              ]
            : [`/prompt-learning/online/jobs/${job.job_id}/events?since_seq=${appState.lastSeq}&limit=200`]

    let payload: any = null
    let lastErr: any = null
    for (const path of paths) {
      try {
        payload = await apiGet(path)
        lastErr = null
        break
      } catch (err: any) {
        lastErr = err
      }
    }

    if (lastErr) {
      if (token !== appState.eventsToken || snapshot.selectedJob?.job_id !== jobId) {
        return true
      }
      snapshot.lastError = lastErr?.message || "Failed to load events"
      return false
    }

    if (token !== appState.eventsToken || snapshot.selectedJob?.job_id !== jobId) {
      return true
    }

    const { events, nextSeq } = extractEvents(payload)
    if (events.length > 0) {
      snapshot.events.push(...events)
      const filter = appState.eventFilter.trim().toLowerCase()
      const newMatchCount =
        filter.length === 0 ? events.length : events.filter((event) => eventMatchesFilter(event, filter)).length

      if (appState.activePane === "events" && newMatchCount > 0) {
        if (appState.selectedEventIndex > 0) {
          appState.selectedEventIndex += newMatchCount
        }
        if (appState.eventWindowStart > 0) {
          appState.eventWindowStart += newMatchCount
        }
      }

      if (config.eventHistoryLimit > 0 && snapshot.events.length > config.eventHistoryLimit) {
        snapshot.events = snapshot.events.slice(-config.eventHistoryLimit)
        appState.selectedEventIndex = clamp(
          appState.selectedEventIndex,
          0,
          Math.max(0, snapshot.events.length - 1),
        )
        appState.eventWindowStart = clamp(
          appState.eventWindowStart,
          0,
          Math.max(0, snapshot.events.length - Math.max(1, config.eventVisibleCount)),
        )
      }
      appState.lastSeq = Math.max(appState.lastSeq, ...events.map((e) => e.seq))
    }

    if (typeof nextSeq === "number" && Number.isFinite(nextSeq)) {
      appState.lastSeq = Math.max(appState.lastSeq, nextSeq)
    }

    return true
  } catch {
    return false
  }
}

