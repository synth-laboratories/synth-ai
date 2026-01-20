/**
 * Job fetching and selection operations.
 */
import type { AppContext } from "../context"
import { extractJobs, mergeJobs, coerceJob, isEvalJob, type JobSummary } from "../tui_data"
import { apiGet } from "./client"
import { fetchCandidatesForJob, candidatesToLegacyFormat } from "./candidates"

function extractBestSnapshotId(payload: any): string | null {
  if (!payload) return null
  // Check multiple possible locations for the best snapshot ID
  return (
    payload.best_snapshot_id ||
    payload.prompt_best_snapshot_id ||
    payload.best_snapshot?.id ||
    payload.metadata?.prompt_best_snapshot_id ||
    payload.metadata?.best_snapshot_id ||
    null
  )
}

export async function refreshJobs(ctx: AppContext): Promise<boolean> {
  const { snapshot, appState, config } = ctx.state

  try {
    snapshot.status = "Refreshing jobs..."
    // Prefer unified jobs endpoint (includes prompt-learning + learning/eval).
    // Fallback to legacy split fetch if unavailable.
    let jobs: JobSummary[] = []
    let learningError: string | null = null
    try {
      const unifiedPayload = await apiGet(`/jobs?limit=${config.jobLimit}`)
      jobs = extractJobs(unifiedPayload)
    } catch (err: any) {
      // Legacy behavior
      const promptPayload = await apiGet(`/prompt-learning/online/jobs?limit=${config.jobLimit}&offset=0`)
      const promptJobs = extractJobs(promptPayload, "prompt-learning")

      let learningJobs: JobSummary[] = []
      try {
        const learningPayload = await apiGet(`/learning/jobs?limit=${config.jobLimit}`)
        learningJobs = extractJobs(learningPayload, "learning")
      } catch (err2: any) {
        learningError = err2?.message || "Failed to load learning jobs"
      }

      jobs = mergeJobs(promptJobs, learningJobs)
    }

    snapshot.jobs = jobs
    snapshot.lastRefresh = Date.now()
    snapshot.lastError = learningError

    if (!snapshot.selectedJob && jobs.length > 0 && !appState.autoSelected) {
      appState.autoSelected = true
      await selectJob(ctx, jobs[0].job_id)
      return true
    }

    if (snapshot.selectedJob) {
      const match = jobs.find((j) => j.job_id === snapshot.selectedJob?.job_id)
      if (match && !snapshot.selectedJob.metadata) {
        snapshot.selectedJob = match
      }
    }

    if (jobs.length === 0) {
      snapshot.status = "No jobs found"
    } else {
      snapshot.status = `Loaded ${jobs.length} job(s)`
    }
    return true
  } catch (err: any) {
    snapshot.lastError = err?.message || "Failed to load jobs"
    snapshot.status = "Failed to load jobs"
    return false
  }
}

export async function selectJob(ctx: AppContext, jobId: string): Promise<void> {
  const { snapshot, appState } = ctx.state

  const token = ++appState.jobSelectToken
  appState.eventsToken++
  appState.lastSeq = 0
  snapshot.events = []
  snapshot.metrics = {}
  snapshot.bestSnapshotId = null
  snapshot.bestSnapshot = null
  snapshot.evalSummary = null
  snapshot.evalResultRows = []
  snapshot.allCandidates = []
  snapshot.apiCandidates = []
  snapshot.apiCandidatesLoaded = false
  appState.selectedEventIndex = 0
  appState.eventWindowStart = 0

  const immediate = snapshot.jobs.find((job) => job.job_id === jobId)
  snapshot.selectedJob =
    immediate ??
    ({
      job_id: jobId,
      status: "loading",
      job_type: null,
      created_at: null,
      started_at: null,
      finished_at: null,
      best_reward: null,
      best_snapshot_id: null,
      total_tokens: null,
      total_cost_usd: null,
      error: null,
      job_source: null,
    } as JobSummary)
  snapshot.status = `Loading job ${jobId}...`

  const jobSource = immediate?.job_source ?? null
  try {
    const path =
      jobSource === "eval"
        ? `/eval/jobs/${jobId}`
        : jobSource === "learning"
          ? `/learning/jobs/${jobId}?include_metadata=true`
          : `/prompt-learning/online/jobs/${jobId}?include_events=false&include_snapshot=false&include_metadata=true`
    const job = await apiGet(path)
    if (token !== appState.jobSelectToken || snapshot.selectedJob?.job_id !== jobId) {
      return
    }

    const coerced = coerceJob(job, jobSource ?? "prompt-learning")
    if (jobSource !== "eval") {
      const jobMeta = job?.metadata ?? {}
      if (job?.prompt_initial_snapshot && !jobMeta.prompt_initial_snapshot) {
        coerced.metadata = { ...jobMeta, prompt_initial_snapshot: job.prompt_initial_snapshot }
      } else {
        coerced.metadata = jobMeta
      }
      snapshot.bestSnapshotId = extractBestSnapshotId(job)
    }
    if (jobSource === "eval" || isEvalJob(coerced)) {
      snapshot.evalSummary = job?.results && typeof job.results === "object" ? job.results : null
    }
    snapshot.selectedJob = coerced
    snapshot.status = `Selected job ${jobId}`
  } catch (err: any) {
    if (token !== appState.jobSelectToken || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    const errMsg = err?.message || `Failed to load job ${jobId}`
    snapshot.lastError = errMsg
    snapshot.status = `Error: ${errMsg}`
  }

  if (jobSource !== "learning" && jobSource !== "eval" && !isEvalJob(snapshot.selectedJob)) {
    await fetchBestSnapshot(ctx, token)
  }
  if (jobSource === "eval" || isEvalJob(snapshot.selectedJob)) {
    await fetchEvalResults(ctx, token)
  }
  
  // Auto-fetch metrics for GEPA jobs
  if (token === appState.jobSelectToken && snapshot.selectedJob) {
    const isGepa =
      snapshot.selectedJob.job_type === "gepa" ||
      snapshot.selectedJob.job_type === "graph_gepa" ||
      snapshot.selectedJob.job_type === "graph_evolve"
    if (isGepa) {
      // Small delay to ensure job data is fully loaded
      await new Promise(resolve => setTimeout(resolve, 100))
      if (token === appState.jobSelectToken && snapshot.selectedJob?.job_id === jobId) {
        await fetchMetrics(ctx)
      }
    }
  }

  // Fetch candidates from the unified candidates API (non-blocking)
  // This supplements event-based candidate tracking with persisted data
  if (token === appState.jobSelectToken && snapshot.selectedJob?.job_id === jobId) {
    fetchApiCandidates(ctx, token).catch(() => {
      // Silently ignore - API candidates are supplementary
    })
  }
}

export async function fetchBestSnapshot(ctx: AppContext, token?: number): Promise<void> {
  const { snapshot, appState } = ctx.state
  const job = snapshot.selectedJob
  if (!job) return

  const jobId = job.job_id
  const snapshotId = snapshot.bestSnapshotId

  try {
    let payload: any
    // If we have a snapshot ID, use the specific snapshot endpoint
    if (snapshotId) {
      payload = await apiGet(`/prompt-learning/online/jobs/${jobId}/snapshots/${snapshotId}`)
      payload = payload?.payload || payload
    } else {
      // Otherwise, use the best-snapshot endpoint which can find it even without an explicit ID
      payload = await apiGet(`/prompt-learning/online/jobs/${jobId}/best-snapshot`)
      // Update bestSnapshotId from the response if it wasn't set
      if (payload?.best_snapshot_id && !snapshot.bestSnapshotId) {
        snapshot.bestSnapshotId = payload.best_snapshot_id
      }
      payload = payload?.best_snapshot || payload
    }

    if ((token != null && token !== appState.jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.bestSnapshot = payload
    snapshot.status = `Loaded best snapshot`
  } catch (err: any) {
    if ((token != null && token !== appState.jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.lastError = err?.message || "Failed to load best snapshot"
  }
}

export async function fetchEvalResults(ctx: AppContext, token?: number): Promise<void> {
  const { snapshot, appState } = ctx.state
  const job = snapshot.selectedJob
  if (!job || !isEvalJob(job)) return

  const jobId = job.job_id
  try {
    snapshot.status = "Loading eval results..."
    const payload = await apiGet(`/eval/jobs/${job.job_id}/results`)
    if ((token != null && token !== appState.jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.evalSummary = payload?.summary && typeof payload.summary === "object" ? payload.summary : null
    snapshot.evalResultRows = Array.isArray(payload?.results) ? payload.results : []
    snapshot.status = `Loaded eval results for ${job.job_id}`
  } catch (err: any) {
    if ((token != null && token !== appState.jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.lastError = err?.message || "Failed to load eval results"
    snapshot.status = "Failed to load eval results"
  }
}

export async function fetchMetrics(ctx: AppContext): Promise<void> {
  const { snapshot } = ctx.state
  const job = snapshot.selectedJob
  if (!job) return

  const jobId = job.job_id
  try {
    if (isEvalJob(job)) {
      await fetchEvalResults(ctx)
      return
    }
    snapshot.status = "Loading metrics..."
    const path =
      job.job_source === "learning"
        ? `/learning/jobs/${job.job_id}/metrics`
        : `/prompt-learning/online/jobs/${job.job_id}/metrics`
    const payload = await apiGet(path)
    if (snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.metrics = payload
    const p: any = payload ?? {}
    const points = Array.isArray(p?.points) ? p.points : []
    
    // Debug: log metrics structure for troubleshooting
    if (points.length === 0) {
      // Always log when no points - helps diagnose the issue
      const gepaMetrics = points.filter((pt: any) => pt?.name?.startsWith("gepa."))
      const allMetricNames = [...new Set(points.map((pt: any) => pt?.name).filter(Boolean))]
      snapshot.status = `No GEPA metrics found (${points.length} total points, ${gepaMetrics.length} gepa metrics, names: ${allMetricNames.slice(0, 5).join(", ") || "none"})`
    } else {
      const gepaMetrics = points.filter((pt: any) => pt?.name?.startsWith("gepa."))
      snapshot.status = `Loaded ${points.length} metric points (${gepaMetrics.length} GEPA metrics) for ${job.job_id}`
    }
  } catch (err: any) {
    if (snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.lastError = err?.message || "Failed to load metrics"
    snapshot.status = "Failed to load metrics"
  }
}

export async function cancelSelected(ctx: AppContext): Promise<void> {
  const { snapshot } = ctx.state
  const job = snapshot.selectedJob
  if (!job) return

  try {
    const { apiPost } = await import("./client")

    // Route to the appropriate cancel endpoint based on job source/type
    let cancelPath: string
    if (job.job_source === "eval" || isEvalJob(job)) {
      cancelPath = `/eval/jobs/${job.job_id}/cancel`
    } else if (job.job_source === "prompt-learning") {
      cancelPath = `/prompt-learning/online/jobs/${job.job_id}/cancel`
    } else {
      // Fallback to unified cancel endpoint (works for graph_evolve, learning, etc.)
      cancelPath = `/jobs/${job.job_id}/cancel`
    }

    const result = await apiPost(cancelPath, {})
    snapshot.status = result?.message || "Cancel requested"
  } catch (err: any) {
    snapshot.lastError = err?.message || "Cancel failed"
    snapshot.status = "Cancel failed"
  }
}

export async function fetchArtifacts(ctx: AppContext): Promise<void> {
  const { snapshot } = ctx.state
  const job = snapshot.selectedJob
  if (!job) return

  try {
    const payload = await apiGet(`/prompt-learning/online/jobs/${job.job_id}/artifacts`)
    snapshot.artifacts = Array.isArray(payload) ? payload : payload?.artifacts || []
    snapshot.status = "Artifacts fetched"
  } catch (err: any) {
    snapshot.lastError = err?.message || "Artifacts fetch failed"
  }
}

/**
 * Fetch candidates from the unified candidates API.
 * This supplements the event-based candidate tracking with persisted candidates.
 */
export async function fetchApiCandidates(ctx: AppContext, token?: number): Promise<void> {
  const { snapshot, appState } = ctx.state
  const job = snapshot.selectedJob
  if (!job) return

  const jobId = job.job_id

  try {
    const response = await fetchCandidatesForJob(jobId, { limit: 500 })

    // Check if job selection changed while fetching
    if ((token != null && token !== appState.jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }

    if (response.candidates.length > 0) {
      // Store raw API candidates
      snapshot.apiCandidates = response.candidates as any[]
      snapshot.apiCandidatesLoaded = true

      // Also merge into allCandidates for compatibility with existing code
      const legacyCandidates = candidatesToLegacyFormat(response.candidates)
      for (const candidate of legacyCandidates) {
        const existing = snapshot.allCandidates.find((c: any) => c.candidate_id === candidate.candidate_id)
        if (!existing) {
          snapshot.allCandidates.push(candidate as any)
        }
      }
    } else {
      snapshot.apiCandidatesLoaded = true
    }
  } catch (err: any) {
    // Non-fatal: API candidates are supplementary
    // The TUI will still work with event-based candidates
    console.error("Failed to fetch API candidates:", err?.message)
    snapshot.apiCandidatesLoaded = true // Mark as loaded even on error
  }
}
