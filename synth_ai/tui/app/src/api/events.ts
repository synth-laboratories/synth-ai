/**
 * Event polling operations.
 */
import type { AppContext } from "../context"
import { extractEvents, isEvalJob, num, type JobEvent } from "../tui_data"
import type { PromptCandidate } from "../types"
import { apiGet } from "./client"
import { isAbortError } from "../utils/abort"
import { isAborted } from "../utils/request"

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
  const { data } = ctx.state
  const { setData } = ctx
  const job = data.selectedJob
  if (!job) return
  
  const trainingType =
    typeof job.training_type === "string" ? job.training_type.toLowerCase() : ""
  const algo = typeof job.algorithm === "string" ? job.algorithm.toLowerCase() : ""
  const metaAlgo =
    isRecord(job.metadata) && typeof job.metadata.algorithm === "string"
      ? job.metadata.algorithm.toLowerCase()
      : ""
  const isGepa =
    trainingType === "gepa" ||
    trainingType === "graph_gepa" ||
    algo === "gepa" ||
    metaAlgo === "gepa"
  if (!isGepa) return
  
  // Only extract if metrics endpoint returned empty
  const metrics: any = data.metrics || {}
  const existingPoints = Array.isArray(metrics?.points) ? metrics.points : []
  if (existingPoints.length > 0) return // Metrics endpoint has data, don't override
  
  // Extract metrics from events and convert to metric points format
  const metricPoints: Array<{ name: string; value: number; step: number; timestamp?: string }> = []
  
  for (const event of events) {
    const data = isRecord(event.data) ? event.data : null
    if (!data) continue
    
    // Extract from prompt.learning.gepa.progress events
    if (event.type === "prompt.learning.gepa.progress") {
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
  
  // If we found metrics in events, add them to the data store.
  if (metricPoints.length > 0) {
    const currentMetrics = isRecord(data.metrics) ? data.metrics : {}
    const currentPoints = Array.isArray((currentMetrics as any).points)
      ? (currentMetrics as any).points
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
    const nextMetrics = {
      ...currentMetrics,
      points: Array.from(byNameAndStep.values()).sort((a, b) => (a.step ?? 0) - (b.step ?? 0)),
    }
    setData("metrics", nextMetrics)
  }
}

function updateCandidatesFromEvents(ctx: AppContext, events: JobEvent[]): void {
  const { data } = ctx.state
  const { setData } = ctx
  if (events.length === 0) return

  const nextCandidates = (data.allCandidates ?? []).map((candidate) => ({
    ...candidate,
    payload: isRecord(candidate.payload) ? { ...candidate.payload } : candidate.payload,
  }))
  const byId = new Map<string, number>()
  for (let idx = 0; idx < nextCandidates.length; idx++) {
    byId.set(nextCandidates[idx].id, idx)
  }

  for (const event of events) {
    if (event.type === "prompt.learning.gepa.frontier_updated") {
      const eventData = isRecord(event.data) ? event.data : null
      const frontier = Array.isArray(eventData?.frontier) ? eventData?.frontier : []
      const frontierScores = isRecord(eventData?.frontier_scores) ? eventData?.frontier_scores : null
      const frontierSet = new Set(frontier.map((id) => String(id)))
      for (let idx = 0; idx < nextCandidates.length; idx++) {
        const candidate = nextCandidates[idx]
        const payload = isRecord(candidate.payload) ? candidate.payload : {}
        const nextPayload = { ...payload, is_pareto: frontierSet.has(candidate.id) }
        const nextReward =
          frontierScores && frontierScores[candidate.id] != null
            ? num(frontierScores[candidate.id]) ?? candidate.reward
            : candidate.reward
        nextCandidates[idx] = {
          ...candidate,
          reward: nextReward,
          payload: nextPayload,
        }
      }
      continue
    }

    const candidate = extractCandidateFromEvent(event)
    if (!candidate) continue
    const existingIdx = byId.get(candidate.id)
    if (existingIdx != null) {
      const existing = nextCandidates[existingIdx]
      const existingPayload = isRecord(existing.payload) ? existing.payload : {}
      const nextPayload = {
        ...existingPayload,
        ...(isRecord(candidate.payload) ? candidate.payload : {}),
      }
      nextCandidates[existingIdx] = {
        ...existing,
        reward: candidate.reward ?? existing.reward,
        payload: nextPayload,
        tag: candidate.tag,
        createdAt: existing.createdAt || candidate.createdAt,
        isBaseline: existing.isBaseline || candidate.isBaseline,
      }
    } else {
      byId.set(candidate.id, nextCandidates.length)
      nextCandidates.push(candidate)
    }
  }

  setData("allCandidates", nextCandidates)
}

export async function refreshEvents(
  ctx: AppContext,
): Promise<boolean> {
  const { data, ui, config } = ctx.state
  const { setData, setUi } = ctx
  const job = data.selectedJob
  if (!job) return true

  const jobId = job.job_id
  const token = ui.eventsToken
  let nextLastSeq = ui.lastSeq

  try {
    const isGepa = job.training_type === "gepa" || job.training_type === "graph_gepa"
    const paths =
      isEvalJob(job)
        ? [
            `/eval/jobs/${job.job_id}/events?since_seq=${ui.lastSeq}&limit=200`,
            `/learning/jobs/${job.job_id}/events?since_seq=${ui.lastSeq}&limit=200`,
          ]
        : job.job_source === "learning"
          ? [`/learning/jobs/${job.job_id}/events?since_seq=${ui.lastSeq}&limit=200`]
          : isGepa
            ? [
                `/prompt-learning/online/jobs/${job.job_id}/events?since_seq=${ui.lastSeq}&limit=200`,
                `/learning/jobs/${job.job_id}/events?since_seq=${ui.lastSeq}&limit=200`,
              ]
            : [`/prompt-learning/online/jobs/${job.job_id}/events?since_seq=${ui.lastSeq}&limit=200`]

    let payload: any = null
    let lastErr: any = null
    for (const path of paths) {
      try {
        if (isAborted()) return true
        payload = await apiGet(path)
        lastErr = null
        break
      } catch (err: any) {
        if (isAbortError(err)) return true
        lastErr = err
      }
    }

    if (lastErr) {
      if (token !== ctx.state.ui.eventsToken || ctx.state.data.selectedJob?.job_id !== jobId) {
        return true
      }
      setData("lastError", lastErr?.message || "Failed to load events")
      return false
    }

    if (token !== ctx.state.ui.eventsToken || ctx.state.data.selectedJob?.job_id !== jobId) {
      return true
    }

    const { events, nextSeq } = extractEvents(payload)
    if (events.length > 0) {
      // Deduplicate by seq to be resilient to overlapping polling/SSE/backfills.
      const existingEvents = data.events
      const existingSeqs = new Set(existingEvents.map((e) => e.seq))
      const newEvents = events.filter((e) => !existingSeqs.has(e.seq))
      if (newEvents.length === 0) {
        // Still advance lastSeq if the server tells us to.
        if (typeof nextSeq === "number" && Number.isFinite(nextSeq)) {
          nextLastSeq = Math.max(nextLastSeq, nextSeq)
        }
        if (nextLastSeq !== ui.lastSeq) {
          setUi("lastSeq", nextLastSeq)
        }
        return true
      }

      let mergedEvents = [...existingEvents, ...newEvents]
      updateCandidatesFromEvents(ctx, newEvents)
      
      // Extract GEPA metrics from events as fallback if metrics endpoint is empty
      extractGEPAMetricsFromEvents(ctx, newEvents)
      
      if (config.eventHistoryLimit > 0 && mergedEvents.length > config.eventHistoryLimit) {
        mergedEvents = mergedEvents.slice(-config.eventHistoryLimit)
      }
      setData("events", mergedEvents)
      nextLastSeq = Math.max(nextLastSeq, ...newEvents.map((e) => e.seq))
    }

    if (typeof nextSeq === "number" && Number.isFinite(nextSeq)) {
      nextLastSeq = Math.max(nextLastSeq, nextSeq)
    }

    if (nextLastSeq !== ui.lastSeq) {
      setUi("lastSeq", nextLastSeq)
    }

    return true
  } catch (err: any) {
    if (isAbortError(err)) return true
    return false
  }
}
