/**
 * Event polling operations.
 */
import type { AppContext } from "../context"
import { extractEvents, isEvalJob, num, type JobEvent } from "../tui_data"
import type { PromptCandidate } from "../types"
import { apiGet } from "./client"
import { isAbortError } from "../utils/abort"

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

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

function extractCandidateFromEvent(event: JobEvent): PromptCandidate | null {
  const data = event.data
  if (!isRecord(data)) return null

  const candidatePayload =
    (isRecord(data.program_candidate) && data.program_candidate) ||
    (isRecord(data.candidate) && data.candidate) ||
    data
  if (!isRecord(candidatePayload)) return null

  const candidateId =
    candidatePayload.candidate_id ||
    candidatePayload.version_id ||
    candidatePayload.id ||
    null
  if (!candidateId) return null

  const score = isRecord(candidatePayload.score) ? candidatePayload.score : null
  const reward =
    num(
      candidatePayload.reward ??
        candidatePayload.accuracy ??
        candidatePayload.full_score ??
        candidatePayload.minibatch_score ??
        score?.reward ??
        score?.accuracy,
    ) ?? null

  const mutationType =
    candidatePayload.mutation_type || candidatePayload.operator || candidatePayload.mutation || null
  const isBaseline =
    mutationType === "baseline" ||
    mutationType === "initial_population" ||
    candidatePayload.is_baseline === true

  return {
    id: String(candidateId),
    isBaseline,
    reward,
    payload: candidatePayload,
    createdAt: event.timestamp ?? null,
    tag: event.type,
  }
}

function extractGEPAMetricsFromEvents(ctx: AppContext, events: JobEvent[]): void {
  const { snapshot } = ctx.state
  const job = snapshot.selectedJob
  if (!job) return
  
  const isGepa =
    job.job_type === "gepa" ||
    job.job_type === "graph_gepa" ||
    job.job_type === "graph_evolve"
  if (!isGepa) return
  
  // Only extract if metrics endpoint returned empty
  const metrics: any = snapshot.metrics || {}
  const existingPoints = Array.isArray(metrics?.points) ? metrics.points : []
  if (existingPoints.length > 0) return // Metrics endpoint has data, don't override
  
  // Extract metrics from events and convert to metric points format
  const metricPoints: Array<{ name: string; value: number; step: number; timestamp?: string }> = []
  
  for (const event of events) {
    const data = isRecord(event.data) ? event.data : null
    if (!data) continue
    
    // Extract from GEPA progress events (canonical format)
    if (event.type === "learning.policy.gepa.job.progress") {
      const step = num(data.generation) ?? num(data.candidates_evaluated) ?? 0
      if (typeof data.frontier_density === "number") {
        metricPoints.push({
          name: "gepa.frontier.density",
          value: data.frontier_density,
          step: step,
          timestamp: event.timestamp ?? undefined,
        })
      }
      if (typeof data.total_seeds_solved === "number") {
        metricPoints.push({
          name: "gepa.frontier.total_seeds_solved",
          value: data.total_seeds_solved,
          step: step,
          timestamp: event.timestamp ?? undefined,
        })
      }
      if (isRecord(data.pareto_growth)) {
        const growth = data.pareto_growth
        if (typeof growth.all_time === "number") {
          metricPoints.push({
            name: "gepa.pareto.growth.all_time",
            value: growth.all_time,
            step: step,
            timestamp: event.timestamp ?? undefined,
          })
        }
      }
    }
    
    // Extract from prompt.learning.gepa.archive.frontier_improved events
    if (event.type === "prompt.learning.gepa.archive.frontier_improved" || 
        event.type === "prompt.learning.gepa.frontier_updated") {
      const step = num(data.generation) ?? num(data.candidates_evaluated) ?? 0
      if (typeof data.frontier_density === "number") {
        metricPoints.push({
          name: "gepa.frontier.density",
          value: data.frontier_density,
          step: step,
          timestamp: event.timestamp ?? undefined,
        })
      }
      if (typeof data.total_seeds_solved === "number") {
        metricPoints.push({
          name: "gepa.frontier.total_seeds_solved",
          value: data.total_seeds_solved,
          step: step,
          timestamp: event.timestamp ?? undefined,
        })
      }
      if (isRecord(data.pareto_growth)) {
        const growth = data.pareto_growth
        if (typeof growth.all_time === "number") {
          metricPoints.push({
            name: "gepa.pareto.growth.all_time",
            value: growth.all_time,
            step: step,
            timestamp: event.timestamp ?? undefined,
          })
        }
      }
    }
  }
  
  // If we found metrics in events, add them to snapshot.metrics
  if (metricPoints.length > 0) {
    if (!snapshot.metrics || typeof snapshot.metrics !== "object") {
      snapshot.metrics = { points: [] }
    }
    const currentPoints = Array.isArray((snapshot.metrics as any).points) 
      ? (snapshot.metrics as any).points 
      : []
    // Merge, keeping latest by step for each metric name
    const byNameAndStep = new Map<string, any>()
    for (const pt of [...currentPoints, ...metricPoints]) {
      const key = `${pt.name}:${pt.step}`
      const existing = byNameAndStep.get(key)
      if (!existing || (pt.timestamp && existing.timestamp && pt.timestamp > existing.timestamp)) {
        byNameAndStep.set(key, pt)
      }
    }
    ;(snapshot.metrics as any).points = Array.from(byNameAndStep.values())
      .sort((a, b) => (a.step ?? 0) - (b.step ?? 0))
  }
}

function updateCandidatesFromEvents(ctx: AppContext, events: JobEvent[]): void {
  const { snapshot } = ctx.state
  if (events.length === 0) return

  const byId = new Map<string, PromptCandidate>()
  for (const candidate of snapshot.allCandidates) {
    byId.set(candidate.id, candidate)
  }

  for (const event of events) {
    if (event.type === "learning.policy.gepa.frontier.updated") {
      const data = isRecord(event.data) ? event.data : null
      const frontier = Array.isArray(data?.frontier) ? data?.frontier : []
      const frontierScores = isRecord(data?.frontier_scores) ? data?.frontier_scores : null
      const frontierSet = new Set(frontier.map((id) => String(id)))
      for (const candidate of snapshot.allCandidates) {
        candidate.payload = { ...candidate.payload, is_pareto: frontierSet.has(candidate.id) }
        if (frontierScores && frontierScores[candidate.id] != null) {
          candidate.reward = num(frontierScores[candidate.id]) ?? candidate.reward
        }
      }
      continue
    }

    const candidate = extractCandidateFromEvent(event)
    if (!candidate) continue
    const existing = byId.get(candidate.id)
    if (existing) {
      if (candidate.reward != null) existing.reward = candidate.reward
      existing.payload = { ...existing.payload, ...candidate.payload }
      existing.tag = candidate.tag
      existing.createdAt = existing.createdAt || candidate.createdAt
      existing.isBaseline = existing.isBaseline || candidate.isBaseline
    } else {
      byId.set(candidate.id, candidate)
      snapshot.allCandidates.push(candidate)
    }
  }
}

export async function refreshEvents(
  ctx: AppContext,
  options: { signal?: AbortSignal } = {},
): Promise<boolean> {
  const { snapshot, appState, config } = ctx.state
  const job = snapshot.selectedJob
  if (!job) return true

  const jobId = job.job_id
  const token = appState.eventsToken

  try {
    const isGepa = job.job_type === "gepa" || job.job_type === "graph_gepa"
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
        if (options.signal?.aborted) return true
        payload = await apiGet(path, options)
        lastErr = null
        break
      } catch (err: any) {
        if (isAbortError(err)) return true
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
      // Deduplicate by seq to be resilient to overlapping polling/SSE/backfills.
      const existingSeqs = new Set(snapshot.events.map((e) => e.seq))
      const newEvents = events.filter((e) => !existingSeqs.has(e.seq))
      if (newEvents.length === 0) {
        // Still advance lastSeq if the server tells us to.
        if (typeof nextSeq === "number" && Number.isFinite(nextSeq)) {
          appState.lastSeq = Math.max(appState.lastSeq, nextSeq)
        }
        return true
      }

      snapshot.events.push(...newEvents)
      updateCandidatesFromEvents(ctx, newEvents)
      
      // Extract GEPA metrics from events as fallback if metrics endpoint is empty
      extractGEPAMetricsFromEvents(ctx, newEvents)
      
      const filter = appState.eventFilter.trim().toLowerCase()
      const newMatchCount =
        filter.length === 0
          ? newEvents.length
          : newEvents.filter((event) => eventMatchesFilter(event, filter)).length

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
      appState.lastSeq = Math.max(appState.lastSeq, ...newEvents.map((e) => e.seq))
    }

    if (typeof nextSeq === "number" && Number.isFinite(nextSeq)) {
      appState.lastSeq = Math.max(appState.lastSeq, nextSeq)
    }

    return true
  } catch (err: any) {
    if (isAbortError(err)) return true
    return false
  }
}
