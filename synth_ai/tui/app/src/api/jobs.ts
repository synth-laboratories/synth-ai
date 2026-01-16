/**
 * Job fetching and selection operations.
 */
import type { AppContext } from "../context"
import { extractJobs, mergeJobs, sortJobs, coerceJob, isEvalJob, type JobSummary } from "../tui_data"
import { apiGet } from "./client"
import { isAbortError } from "../utils/abort"
import { isAborted } from "../utils/request"
import { log } from "../utils/log"
import { getJobsCacheKey, mergeJobsCache, scheduleJobsCacheWrite } from "../persistence/jobs-cache"

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

function extractEvalMetadata(config: unknown): Record<string, any> {
  if (!isRecord(config)) return {}

  const metadata: Record<string, any> = {}
  const policy = isRecord(config.policy) ? config.policy : {}
  const verifier = isRecord(config.verifier_config) ? config.verifier_config : {}

  if (typeof config.env_name === "string") metadata.env_name = config.env_name
  if (typeof config.app_id === "string") metadata.app_id = config.app_id
  if (typeof config.localapi_url === "string") metadata.localapi_url = config.localapi_url
  else if (typeof config.task_app_url === "string") metadata.localapi_url = config.task_app_url
  if (Array.isArray(config.seeds)) metadata.seeds = config.seeds
  if (typeof config.max_concurrent === "number") metadata.max_concurrent = config.max_concurrent
  if (typeof config.timeout === "number") metadata.timeout = config.timeout
  if (typeof config.mode === "string") metadata.mode = config.mode
  if (typeof config.eval_mode === "string") metadata.eval_mode = config.eval_mode
  if (typeof config.candidate_id === "string") metadata.candidate_id = config.candidate_id
  if (typeof config.candidate_type === "string") metadata.candidate_type = config.candidate_type
  if (typeof config.trace_source_job_id === "string") metadata.trace_source_job_id = config.trace_source_job_id
  if (typeof config.trace_source_type === "string") metadata.trace_source_type = config.trace_source_type
  if (typeof config.task_id === "string") metadata.task_id = config.task_id

  const safePolicy: Record<string, any> = {}
  if (typeof policy.model === "string") safePolicy.model = policy.model
  if (typeof policy.provider === "string") safePolicy.provider = policy.provider
  if (typeof policy.inference_mode === "string") safePolicy.inference_mode = policy.inference_mode
  if (Object.keys(safePolicy).length > 0) metadata.policy = safePolicy

  const safeVerifier: Record<string, any> = {}
  if (typeof verifier.enabled === "boolean") safeVerifier.enabled = verifier.enabled
  if (typeof verifier.reward_source === "string") safeVerifier.reward_source = verifier.reward_source
  if (typeof verifier.backend_model === "string") safeVerifier.backend_model = verifier.backend_model
  if (typeof verifier.backend_provider === "string") safeVerifier.backend_provider = verifier.backend_provider
  if (typeof verifier.verifier_graph_id === "string") safeVerifier.verifier_graph_id = verifier.verifier_graph_id
  if (Object.keys(safeVerifier).length > 0) metadata.verifier_config = safeVerifier

  return metadata
}

function extractBestSnapshotId(payload: any): string | null {
  if (!payload) return null
  // Check multiple possible locations for the best Candidate ID
  return (
    payload.best_snapshot_id ||
    payload.prompt_best_snapshot_id ||
    payload.best_snapshot?.id ||
    payload.metadata?.prompt_best_snapshot_id ||
    payload.metadata?.best_snapshot_id ||
    null
  )
}

const PROMPT_LEARNING_LIMIT_MAX = 200

type RefreshJobsResult = {
  ok: boolean
  promptCount: number
  learningCount: number
  serverCount: number
  serverHasMore: boolean
}

function resolveJobsLimit(ctx: AppContext, override?: number): number {
  const fallback = ctx.state.config.jobLimit
  const value = override ?? ctx.state.ui.jobsListLimit ?? fallback
  const limit = Number.isFinite(value) ? Math.max(1, Number(value)) : fallback
  return Math.max(1, limit)
}

function getJobTimestamp(job: JobSummary): number {
  const value = job.created_at || job.started_at || job.finished_at
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}

export function recordJobsCache(ctx: AppContext, jobs: JobSummary[]): void {
  const { data, ui } = ctx.state
  if (!jobs.length || !data.orgId) return
  const cacheKey = getJobsCacheKey(data.orgId, ui.currentMode)
  const baseCache = data.jobsCacheKey === cacheKey ? data.jobsCache : []
  const nextCache = mergeJobsCache(baseCache, jobs)
  let changed = nextCache.length !== baseCache.length
  if (!changed) {
    for (let i = 0; i < nextCache.length; i += 1) {
      const prev = baseCache[i]
      const next = nextCache[i]
      if (!prev || prev.job_id !== next.job_id || prev.status !== next.status) {
        changed = true
        break
      }
    }
  }
  if (!changed) return

  if (data.jobsCacheKey !== cacheKey) {
    ctx.setData("jobsCacheKey", cacheKey)
    ctx.setData("jobsCacheAppended", [])
  }
  ctx.setData("jobsCache", nextCache)
  scheduleJobsCacheWrite(cacheKey, nextCache)
}

function appendCachedJobs(
  ctx: AppContext,
  options: { batchSize?: number } = {},
): number {
  const { data, config } = ctx.state
  const batchSize = Math.max(1, options.batchSize ?? config.jobLimit)
  if (!data.jobsCache.length) return 0

  const existingIds = new Set(data.jobs.map((job) => job.job_id))
  const oldest = data.jobs[data.jobs.length - 1]
  const oldestTs = oldest ? getJobTimestamp(oldest) : 0
  const candidates = data.jobsCache.filter((job) => job.job_id && !existingIds.has(job.job_id))
  const eligible = oldestTs
    ? candidates.filter((job) => getJobTimestamp(job) <= oldestTs)
    : candidates
  if (!eligible.length) return 0

  const batch = sortJobs(eligible).slice(0, batchSize)
  if (!batch.length) return 0

  ctx.setData("jobsCacheAppended", (prev) => mergeJobs(prev, batch))
  ctx.setData("jobs", (prev) => sortJobs([...prev, ...batch]))
  return batch.length
}

export async function refreshJobs(
  ctx: AppContext,
  options: { limit?: number } = {},
): Promise<RefreshJobsResult> {
  const { data, ui } = ctx.state
  const { setData, setUi } = ctx
  const limit = resolveJobsLimit(ctx, options.limit)
  const promptLimit = Math.min(limit, PROMPT_LEARNING_LIMIT_MAX)
  if (ui.jobsListLimit !== limit) {
    setUi("jobsListLimit", limit)
  }

  try {
    if (isAborted()) {
      return { ok: false, promptCount: 0, learningCount: 0, serverCount: 0, serverHasMore: false }
    }
    setData("status", "Refreshing jobs...")
    const promptPayload = await apiGet(`/prompt-learning/online/jobs?limit=${promptLimit}`)
    const promptJobs = extractJobs(promptPayload, "prompt-learning")

    let learningJobs: JobSummary[] = []
    let learningError: string | null = null
    try {
      if (isAborted()) {
        return { ok: false, promptCount: 0, learningCount: 0, serverCount: 0, serverHasMore: false }
      }
      const learningPayload = await apiGet(`/learning/jobs?limit=${limit}`)
      learningJobs = extractJobs(learningPayload, "learning")
    } catch (err: any) {
      if (isAbortError(err)) {
        return { ok: false, promptCount: 0, learningCount: 0, serverCount: 0, serverHasMore: false }
      }
      learningError = err?.message || "Failed to load learning jobs"
    }

    const serverJobs = mergeJobs(promptJobs, learningJobs)
    const promptHasMore =
      promptJobs.length >= promptLimit && promptLimit < PROMPT_LEARNING_LIMIT_MAX
    const learningHasMore = learningJobs.length >= limit
    const serverHasMore = promptHasMore || learningHasMore

    const serverIds = new Set(serverJobs.map((job) => job.job_id))
    const nextAppended = data.jobsCacheAppended.filter((job) => !serverIds.has(job.job_id))
    const jobs = mergeJobs(serverJobs, nextAppended)

    setData("jobsCacheAppended", nextAppended)
    setData("jobs", jobs)
    setData("lastRefresh", Date.now())
    setData("lastError", learningError)
    setUi("jobsListHasMore", serverHasMore)
    setUi("jobsListServerCount", serverJobs.length)

    if (data.selectedJob) {
      const match = serverJobs.find((j) => j.job_id === data.selectedJob?.job_id)
      if (match && !data.selectedJob.metadata) {
        setData("selectedJob", match)
      }
    }

    if (jobs.length === 0) {
      setData("status", "No jobs found")
    } else {
      setData("status", "")
    }

    recordJobsCache(ctx, serverJobs)
    log("state", "jobs refresh", {
      limit,
      promptCount: promptJobs.length,
      learningCount: learningJobs.length,
      serverCount: serverJobs.length,
      serverHasMore,
      cacheCount: data.jobsCache.length,
    })
    return {
      ok: true,
      promptCount: promptJobs.length,
      learningCount: learningJobs.length,
      serverCount: serverJobs.length,
      serverHasMore,
    }
  } catch (err: any) {
    if (isAbortError(err)) {
      return { ok: false, promptCount: 0, learningCount: 0, serverCount: 0, serverHasMore: false }
    }
    setData("lastError", err?.message || "Failed to load jobs")
    setData("status", "Failed to load jobs")
    return { ok: false, promptCount: 0, learningCount: 0, serverCount: 0, serverHasMore: false }
  }
}

export async function loadMoreJobs(ctx: AppContext): Promise<void> {
  const { ui, config } = ctx.state
  if (ui.jobsListLoadingMore) return

  const step = Math.max(1, config.jobLimit)
  const currentLimit = resolveJobsLimit(ctx)
  const nextLimit = currentLimit + step
  const prevServerCount = ui.jobsListServerCount

  ctx.setUi("jobsListLoadingMore", true)
  ctx.setUi("jobsListLimit", nextLimit)

  try {
    const result = await refreshJobs(ctx, { limit: nextLimit })
    if (!result.ok) return

    const serverGrew = result.serverCount > prevServerCount
    if (!result.serverHasMore && !serverGrew) {
      const appended = appendCachedJobs(ctx, { batchSize: step })
      if (appended > 0) {
        log("state", "jobs load-more cache", { appended })
      }
    }
  } finally {
    ctx.setUi("jobsListLoadingMore", false)
  }
}

export async function selectJob(ctx: AppContext, jobId: string): Promise<void> {
  const { data, ui } = ctx.state
  const { setData, setUi } = ctx

  const token = ui.jobSelectToken + 1
  setUi("jobSelectToken", token)
  setUi("eventsToken", ui.eventsToken + 1)
  setUi("lastSeq", 0)
  setUi("selectedEventId", null)
  setUi("verifierEvolveGenerationIndex", 0)
  setData("events", [])
  setData("metrics", {})
  setData("bestSnapshotId", null)
  setData("bestSnapshot", null)
  setData("evalSummary", null)
  setData("evalResultRows", [])
  setData("allCandidates", [])

  const immediate = data.jobs.find((job) => job.job_id === jobId)
  const placeholder = {
    job_id: jobId,
    status: "loading",
    training_type: null,
    created_at: null,
    started_at: null,
    finished_at: null,
    best_reward: null,
    best_snapshot_id: null,
    total_tokens: null,
    total_cost_usd: null,
    error: null,
    job_source: null,
  } as JobSummary
  setData("selectedJob", immediate ?? placeholder)
  setData("status", `Loading job ${jobId}...`)

  const jobSource = immediate?.job_source ?? null
  try {
    const path =
      jobSource === "eval"
        ? `/eval/jobs/${jobId}`
        : jobSource === "learning"
          ? `/learning/jobs/${jobId}?include_metadata=true`
          : `/prompt-learning/online/jobs/${jobId}?include_events=false&include_snapshot=false&include_metadata=true`
    const job = await apiGet(path)
    if (token !== ctx.state.ui.jobSelectToken || ctx.state.data.selectedJob?.job_id !== jobId) {
      return
    }

    const coerced = coerceJob(job, jobSource ?? "prompt-learning")
    if (jobSource === "eval" || isEvalJob(coerced)) {
      const evalMeta = extractEvalMetadata(job?.config)
      if (Object.keys(evalMeta).length > 0) {
        const existingMeta = isRecord(coerced.metadata) ? coerced.metadata : {}
        coerced.metadata = { ...existingMeta, ...evalMeta }
      }
    }
    if (jobSource !== "eval") {
      const jobMeta = job?.metadata ?? {}
      if (job?.prompt_initial_snapshot && !jobMeta.prompt_initial_snapshot) {
        coerced.metadata = { ...jobMeta, prompt_initial_snapshot: job.prompt_initial_snapshot }
      } else {
        coerced.metadata = jobMeta
      }
      setData("bestSnapshotId", extractBestSnapshotId(job))
    }
    if (jobSource === "eval" || isEvalJob(coerced)) {
      setData("evalSummary", job?.results && typeof job.results === "object" ? job.results : null)
    }
    setData("selectedJob", coerced)
    setData("status", `Selected job ${jobId}`)
  } catch (err: any) {
    if (isAbortError(err)) return
    if (token !== ctx.state.ui.jobSelectToken || ctx.state.data.selectedJob?.job_id !== jobId) {
      return
    }
    const errMsg = err?.message || `Failed to load job ${jobId}`
    setData("lastError", errMsg)
    setData("status", `Error: ${errMsg}`)
  }

  if (jobSource !== "learning" && jobSource !== "eval" && !isEvalJob(ctx.state.data.selectedJob)) {
    await fetchBestSnapshot(ctx, token)
  }
  if (jobSource === "eval" || isEvalJob(ctx.state.data.selectedJob)) {
    await fetchEvalResults(ctx, token)
  }
  if (!isEvalJob(ctx.state.data.selectedJob)) {
    await fetchMetrics(ctx)
  }
  
}

export async function fetchBestSnapshot(
  ctx: AppContext,
  token?: number,
): Promise<void> {
  const { data } = ctx.state
  const { setData } = ctx
  const job = data.selectedJob
  if (!job) return

  const jobId = job.job_id
  const snapshotId = data.bestSnapshotId

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
      if (payload?.best_snapshot_id && !data.bestSnapshotId) {
        setData("bestSnapshotId", payload.best_snapshot_id)
      }
      payload = payload?.best_snapshot || payload
    }

    if ((token != null && token !== ctx.state.ui.jobSelectToken) || ctx.state.data.selectedJob?.job_id !== jobId) {
      return
    }
    setData("bestSnapshot", payload)
    setData("status", "Loaded best Candidate")
  } catch (err: any) {
    if (isAbortError(err)) return
    if ((token != null && token !== ctx.state.ui.jobSelectToken) || ctx.state.data.selectedJob?.job_id !== jobId) {
      return
    }
    setData("lastError", err?.message || "Failed to load best Candidate")
  }
}

export async function fetchEvalResults(
  ctx: AppContext,
  token?: number,
): Promise<void> {
  const { data } = ctx.state
  const { setData } = ctx
  const job = data.selectedJob
  if (!job || !isEvalJob(job)) return

  const jobId = job.job_id
  try {
    setData("status", "Loading eval results...")
    const payload = await apiGet(`/eval/jobs/${job.job_id}/results`)
    if ((token != null && token !== ctx.state.ui.jobSelectToken) || ctx.state.data.selectedJob?.job_id !== jobId) {
      return
    }
    setData("evalSummary", payload?.summary && typeof payload.summary === "object" ? payload.summary : null)
    setData("evalResultRows", Array.isArray(payload?.results) ? payload.results : [])
    setData("status", `Loaded eval results for ${job.job_id}`)
  } catch (err: any) {
    if (isAbortError(err)) return
    if ((token != null && token !== ctx.state.ui.jobSelectToken) || ctx.state.data.selectedJob?.job_id !== jobId) {
      return
    }
    setData("lastError", err?.message || "Failed to load eval results")
    setData("status", "Failed to load eval results")
  }
}

export async function fetchMetrics(ctx: AppContext): Promise<void> {
  const { data } = ctx.state
  const { setData } = ctx
  const job = data.selectedJob
  if (!job) return

  const jobId = job.job_id
  try {
    if (isEvalJob(job)) {
      await fetchEvalResults(ctx, undefined)
      return
    }
    setData("status", "Loading metrics...")
    const path =
      job.job_source === "learning"
        ? `/learning/jobs/${job.job_id}/metrics`
        : `/prompt-learning/online/jobs/${job.job_id}/metrics`
    const payload = await apiGet(path)
    if (ctx.state.data.selectedJob?.job_id !== jobId) {
      return
    }
    setData("metrics", payload)
    setData("status", "")
  } catch (err: any) {
    if (isAbortError(err)) return
    if (ctx.state.data.selectedJob?.job_id !== jobId) {
      return
    }
    setData("lastError", err?.message || "Failed to load metrics")
    setData("status", "Failed to load metrics")
  }
}

export async function cancelSelected(ctx: AppContext): Promise<void> {
  const { data } = ctx.state
  const { setData } = ctx
  const job = data.selectedJob
  if (!job) return

  try {
    const { apiPost } = await import("./client")
    await apiPost(`/prompt-learning/online/jobs/${job.job_id}/cancel`, {})
    setData("status", "Cancel requested")
  } catch (err: any) {
    if (isAbortError(err)) return
    setData("lastError", err?.message || "Cancel failed")
  }
}

export async function fetchArtifacts(ctx: AppContext): Promise<void> {
  const { data } = ctx.state
  const { setData } = ctx
  const job = data.selectedJob
  if (!job) return

  try {
    const payload = await apiGet(`/prompt-learning/online/jobs/${job.job_id}/artifacts`)
    setData("artifacts", Array.isArray(payload) ? payload : payload?.artifacts || [])
    setData("status", "Artifacts fetched")
  } catch (err: any) {
    if (isAbortError(err)) return
    setData("lastError", err?.message || "Artifacts fetch failed")
  }
}
