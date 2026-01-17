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
import {
  cloneJob,
  getJobById,
  getJobsIndex,
  getJobsList,
  replaceJobsIndex,
  upsertJobsIndex,
} from "../state/jobs-index"

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

const JOBS_PAGE_SIZE = 25

type RefreshJobsResult = {
  ok: boolean
  promptCount: number
  learningCount: number
  serverCount: number
  serverHasMore: boolean
}

function resolveJobsLimit(ctx: AppContext, override?: number): number {
  const fallback = JOBS_PAGE_SIZE
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

type DuplicateJobId = {
  id: string
  count: number
  indices: number[]
}

function findDuplicateJobIds(jobs: JobSummary[]): DuplicateJobId[] {
  const counts = new Map<string, DuplicateJobId>()
  jobs.forEach((job, idx) => {
    const id = job.job_id
    if (!id) return
    const entry = counts.get(id)
    if (!entry) {
      counts.set(id, { id, count: 1, indices: [idx] })
    } else {
      entry.count += 1
      entry.indices.push(idx)
    }
  })
  return Array.from(counts.values()).filter((entry) => entry.count > 1)
}

function buildDuplicateKey(duplicates: DuplicateJobId[]): string {
  return duplicates.map((entry) => `${entry.id}:${entry.count}:${entry.indices.join(",")}`).join("|")
}

function countOrderChanges(
  prevIds: string[],
  nextIds: string[],
): { lengthChanged: boolean; changedCount: number } {
  const lengthChanged = prevIds.length !== nextIds.length
  const compareCount = lengthChanged ? Math.min(prevIds.length, nextIds.length) : prevIds.length
  let changedCount = 0
  for (let i = 0; i < compareCount; i += 1) {
    if (prevIds[i] !== nextIds[i]) {
      changedCount += 1
    }
  }
  return { lengthChanged, changedCount }
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

  const existingJobs = getJobsList(data)
  const existingIds = new Set(existingJobs.map((job) => job.job_id))
  const oldest = existingJobs[existingJobs.length - 1]
  const oldestTs = oldest ? getJobTimestamp(oldest) : 0
  const candidates = data.jobsCache.filter((job) => job.job_id && !existingIds.has(job.job_id))
  const eligible = oldestTs
    ? candidates.filter((job) => getJobTimestamp(job) <= oldestTs)
    : candidates
  if (!eligible.length) return 0

  const batch = sortJobs(eligible).slice(0, batchSize)
  if (!batch.length) return 0

  const overlap = batch
    .map((job, idx) => ({ id: job.job_id, idx }))
    .filter((entry) => entry.id && existingIds.has(entry.id))
  if (overlap.length > 0) {
    log("state", "jobs cache append overlap", { overlap })
  }
  const batchDuplicates = findDuplicateJobIds(batch)
  if (batchDuplicates.length > 0) {
    log("state", "jobs cache append duplicates", { duplicates: batchDuplicates })
  }

  ctx.setData("jobsCacheAppended", (prev) => mergeJobs(prev, batch))
  const prevIndex = getJobsIndex(data)
  const nextIndex = upsertJobsIndex(prevIndex, batch)
  const prevIds = prevIndex.order
  const nextIds = nextIndex.order
  const { lengthChanged, changedCount } = countOrderChanges(prevIds, nextIds)
  const prevDuplicates = findDuplicateJobIds(getJobsList(data))
  const nextDuplicates = findDuplicateJobIds(getJobsList({
    jobsById: nextIndex.byId,
    jobsOrder: nextIndex.order,
  }))
  log("state", "jobs cache append apply", {
    batchCount: batch.length,
    batchIds: batch.map((job) => job.job_id),
    prevLength: prevIds.length,
    nextLength: nextIds.length,
    changedCount: lengthChanged ? null : changedCount,
    prevDuplicates,
    nextDuplicates,
    prevHead: prevIds.slice(0, Math.min(12, prevIds.length)),
    nextHead: nextIds.slice(0, Math.min(12, nextIds.length)),
    prevTail: prevIds.slice(-Math.min(12, prevIds.length)),
    nextTail: nextIds.slice(-Math.min(12, nextIds.length)),
    prevIds,
    nextIds,
  })
  ctx.setData({ jobsById: nextIndex.byId, jobsOrder: nextIndex.order })
  return batch.length
}

export async function refreshJobs(
  ctx: AppContext,
  options: { limit?: number } = {},
): Promise<RefreshJobsResult> {
  const { data, ui } = ctx.state
  const { setData, setUi } = ctx
  const limit = resolveJobsLimit(ctx, options.limit)
  const promptLimit = limit
  const start = Date.now()
  let promptMs: number | null = null
  let promptExtractMs: number | null = null
  let learningMs: number | null = null
  let learningExtractMs: number | null = null
  let mergeMs: number | null = null
  if (ui.jobsListLimit !== limit) {
    setUi("jobsListLimit", limit)
  }

  try {
    if (isAborted()) {
      return { ok: false, promptCount: 0, learningCount: 0, serverCount: 0, serverHasMore: false }
    }
    setData("status", "Refreshing jobs...")
    const learningPromise = (async () => {
      const learningStart = Date.now()
      try {
        if (isAborted()) {
          return { jobs: [] as JobSummary[], error: null as string | null, aborted: true }
        }
        const learningPayload = await apiGet(`/learning/jobs?limit=${limit}`)
        learningMs = Date.now() - learningStart
        const extractStart = Date.now()
        return {
          jobs: extractJobs(learningPayload, "learning"),
          error: null,
          aborted: false,
          extractMs: Date.now() - extractStart,
        }
      } catch (err: any) {
        learningMs = Date.now() - learningStart
        if (isAbortError(err)) {
          return { jobs: [] as JobSummary[], error: null as string | null, aborted: true }
        }
        return {
          jobs: [] as JobSummary[],
          error: err?.message || "Failed to load learning jobs",
          aborted: false,
        }
      }
    })()

    const promptStart = Date.now()
    const promptPayload = await apiGet(`/prompt-learning/online/jobs?limit=${promptLimit}`)
    promptMs = Date.now() - promptStart
    const promptExtractStart = Date.now()
    const promptJobs = extractJobs(promptPayload, "prompt-learning")
    promptExtractMs = Date.now() - promptExtractStart
    const promptDuplicates = findDuplicateJobIds(promptJobs)
    if (promptDuplicates.length > 0) {
      log("state", "jobs refresh duplicates", { source: "prompt", duplicates: promptDuplicates })
    }

    const learningResult = await learningPromise
    if (learningResult.aborted) {
      return { ok: false, promptCount: 0, learningCount: 0, serverCount: 0, serverHasMore: false }
    }
    const learningJobs = learningResult.jobs
    learningExtractMs = learningResult.extractMs ?? learningExtractMs
    const learningError = learningResult.error
    const learningDuplicates = findDuplicateJobIds(learningJobs)
    if (learningDuplicates.length > 0) {
      log("state", "jobs refresh duplicates", { source: "learning", duplicates: learningDuplicates })
    }

    const mergeStart = Date.now()
    const serverJobs = mergeJobs(promptJobs, learningJobs)
    const serverDuplicates = findDuplicateJobIds(serverJobs)
    if (serverDuplicates.length > 0) {
      log("state", "jobs refresh duplicates", { source: "merged", duplicates: serverDuplicates })
    }
    const promptHasMore = promptJobs.length >= promptLimit
    const learningHasMore = learningJobs.length >= limit
    const serverHasMore = promptHasMore || learningHasMore

    const serverIds = new Set(serverJobs.map((job) => job.job_id))
    const nextAppended = data.jobsCacheAppended.filter((job) => !serverIds.has(job.job_id))
    const appendedDuplicates = findDuplicateJobIds(nextAppended)
    if (appendedDuplicates.length > 0) {
      log("state", "jobs refresh duplicates", { source: "appended", duplicates: appendedDuplicates })
    }
    const jobs = mergeJobs(serverJobs, nextAppended)
    const finalDuplicates = findDuplicateJobIds(jobs)
    if (finalDuplicates.length > 0) {
      log("state", "jobs refresh duplicates", { source: "final", duplicates: finalDuplicates })
    }
    mergeMs = Date.now() - mergeStart

    const prevIds = data.jobsOrder
    const nextIndex = replaceJobsIndex(jobs)
    const nextIds = nextIndex.order
    const { lengthChanged, changedCount } = countOrderChanges(prevIds, nextIds)
    const prevDuplicates = findDuplicateJobIds(getJobsList(data))
    const nextDuplicates = findDuplicateJobIds(getJobsList({
      jobsById: nextIndex.byId,
      jobsOrder: nextIndex.order,
    }))
    const duplicatesChanged = buildDuplicateKey(prevDuplicates) !== buildDuplicateKey(nextDuplicates)
    const orderChanged = lengthChanged || changedCount > 0
    if (orderChanged || duplicatesChanged) {
      log("state", "jobs refresh apply", {
        prevLength: prevIds.length,
        nextLength: nextIds.length,
        changedCount: lengthChanged ? null : changedCount,
        prevDuplicates,
        nextDuplicates,
        prevHead: prevIds.slice(0, Math.min(12, prevIds.length)),
        nextHead: nextIds.slice(0, Math.min(12, nextIds.length)),
        prevTail: prevIds.slice(-Math.min(12, prevIds.length)),
        nextTail: nextIds.slice(-Math.min(12, nextIds.length)),
        prevIds,
        nextIds,
        selectedJobId: data.selectedJob?.job_id ?? null,
      })
    }

    setData("jobsCacheAppended", nextAppended)
    setData({ jobsById: nextIndex.byId, jobsOrder: nextIndex.order })
    setData("lastRefresh", Date.now())
    setData("lastError", learningError)
    setUi("jobsListHasMore", serverHasMore)
    setUi("jobsListServerCount", serverJobs.length)

    if (data.selectedJob) {
      const match = nextIndex.byId[data.selectedJob?.job_id ?? ""]
      if (match && !data.selectedJob.metadata) {
        setData("selectedJob", cloneJob(match))
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
      promptMs,
      promptExtractMs,
      learningMs,
      learningExtractMs,
      mergeMs,
      totalMs: Date.now() - start,
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
  const { ui } = ctx.state
  if (ui.jobsListLoadingMore) return

  const step = JOBS_PAGE_SIZE
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
  const selectStart = Date.now()

  const token = ui.jobSelectToken + 1
  const immediate = getJobById(data, jobId)
  log("state", "selectJob start", {
    jobId,
    cached: Boolean(immediate),
    jobSource: immediate?.job_source ?? null,
    token,
  })
  const resetStart = Date.now()
  setUi("jobSelectToken", token)
  setUi("eventsToken", ui.eventsToken + 1)
  setUi("lastSeq", 0)
  setUi("selectedEventId", null)
  setUi("verifierEvolveGenerationIndex", 0)
  log("state", "selectJob reset ui", {
    jobId,
    token,
    totalMs: Date.now() - resetStart,
  })

  const dataResetStart = Date.now()
  setData("events", [])
  setData("metrics", {})
  setData("bestSnapshotId", null)
  setData("bestSnapshot", null)
  setData("evalSummary", null)
  setData("evalResultRows", [])
  setData("allCandidates", [])
  log("state", "selectJob reset data", {
    jobId,
    token,
    totalMs: Date.now() - dataResetStart,
  })
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
  const selectionResetStart = Date.now()
  setData("selectedJob", immediate ? cloneJob(immediate) : placeholder)
  setData("status", `Loading job ${jobId}...`)
  log("state", "selectJob reset selection", {
    jobId,
    token,
    totalMs: Date.now() - selectionResetStart,
  })

  const jobSource = immediate?.job_source ?? null
  try {
    const detailsStart = Date.now()
    const path =
      jobSource === "eval"
        ? `/eval/jobs/${jobId}`
        : jobSource === "learning"
          ? `/learning/jobs/${jobId}?include_metadata=true`
          : `/prompt-learning/online/jobs/${jobId}?include_events=false&include_snapshot=false&include_metadata=true`
    const job = await apiGet(path)
    const detailsMs = Date.now() - detailsStart
    if (token !== ctx.state.ui.jobSelectToken || ctx.state.data.selectedJob?.job_id !== jobId) {
      log("state", "selectJob stale", {
        jobId,
        token,
        currentToken: ctx.state.ui.jobSelectToken,
        detailsMs,
        stage: "details",
      })
      return
    }

    const processStart = Date.now()
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
    log("state", "selectJob details", {
      jobId,
      jobSource: coerced.job_source ?? jobSource ?? null,
      status: coerced.status,
      detailsMs,
      processMs: Date.now() - processStart,
      totalMs: Date.now() - selectStart,
    })
  } catch (err: any) {
    if (isAbortError(err)) return
    if (token !== ctx.state.ui.jobSelectToken || ctx.state.data.selectedJob?.job_id !== jobId) {
      log("state", "selectJob stale", {
        jobId,
        token,
        currentToken: ctx.state.ui.jobSelectToken,
        stage: "details-error",
        error: err?.message || "unknown",
        totalMs: Date.now() - selectStart,
      })
      return
    }
    const errMsg = err?.message || `Failed to load job ${jobId}`
    setData("lastError", errMsg)
    setData("status", `Error: ${errMsg}`)
    log("state", "selectJob error", {
      jobId,
      jobSource,
      error: errMsg,
      totalMs: Date.now() - selectStart,
    })
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
  log("state", "selectJob complete", {
    jobId,
    jobSource: ctx.state.data.selectedJob?.job_source ?? jobSource ?? null,
    totalMs: Date.now() - selectStart,
  })
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
  const start = Date.now()
  const endpoint = snapshotId
    ? `/prompt-learning/online/jobs/${jobId}/snapshots/${snapshotId}`
    : `/prompt-learning/online/jobs/${jobId}/best-snapshot`

  try {
    let payload: any
    // If we have a snapshot ID, use the specific snapshot endpoint
    if (snapshotId) {
      payload = await apiGet(endpoint)
      payload = payload?.payload || payload
    } else {
      // Otherwise, use the best-snapshot endpoint which can find it even without an explicit ID
      payload = await apiGet(endpoint)
      // Update bestSnapshotId from the response if it wasn't set
      if (payload?.best_snapshot_id && !data.bestSnapshotId) {
        setData("bestSnapshotId", payload.best_snapshot_id)
      }
      payload = payload?.best_snapshot || payload
    }

    if ((token != null && token !== ctx.state.ui.jobSelectToken) || ctx.state.data.selectedJob?.job_id !== jobId) {
      log("state", "bestSnapshot stale", {
        jobId,
        endpoint,
        token,
        currentToken: ctx.state.ui.jobSelectToken,
        totalMs: Date.now() - start,
      })
      return
    }
    setData("bestSnapshot", payload)
    setData("status", "Loaded best Candidate")
    log("state", "bestSnapshot fetched", {
      jobId,
      endpoint,
      hasSnapshotId: Boolean(snapshotId),
      payloadKeys: payload && typeof payload === "object" ? Object.keys(payload).length : 0,
      totalMs: Date.now() - start,
    })
  } catch (err: any) {
    if (isAbortError(err)) return
    if ((token != null && token !== ctx.state.ui.jobSelectToken) || ctx.state.data.selectedJob?.job_id !== jobId) {
      log("state", "bestSnapshot stale", {
        jobId,
        endpoint,
        token,
        currentToken: ctx.state.ui.jobSelectToken,
        error: err?.message || "unknown",
        totalMs: Date.now() - start,
      })
      return
    }
    setData("lastError", err?.message || "Failed to load best Candidate")
    log("state", "bestSnapshot error", {
      jobId,
      endpoint,
      error: err?.message || "unknown",
      totalMs: Date.now() - start,
    })
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
  const start = Date.now()
  try {
    setData("status", "Loading eval results...")
    const payload = await apiGet(`/eval/jobs/${job.job_id}/results`)
    if ((token != null && token !== ctx.state.ui.jobSelectToken) || ctx.state.data.selectedJob?.job_id !== jobId) {
      log("state", "evalResults stale", {
        jobId,
        token,
        currentToken: ctx.state.ui.jobSelectToken,
        totalMs: Date.now() - start,
      })
      return
    }
    const summary = payload?.summary && typeof payload.summary === "object" ? payload.summary : null
    const rows = Array.isArray(payload?.results) ? payload.results : []
    setData("evalSummary", summary)
    setData("evalResultRows", rows)
    setData("status", `Loaded eval results for ${job.job_id}`)
    log("state", "evalResults fetched", {
      jobId,
      summaryKeys: summary ? Object.keys(summary).length : 0,
      rows: rows.length,
      totalMs: Date.now() - start,
    })
  } catch (err: any) {
    if (isAbortError(err)) return
    if ((token != null && token !== ctx.state.ui.jobSelectToken) || ctx.state.data.selectedJob?.job_id !== jobId) {
      log("state", "evalResults stale", {
        jobId,
        token,
        currentToken: ctx.state.ui.jobSelectToken,
        error: err?.message || "unknown",
        totalMs: Date.now() - start,
      })
      return
    }
    setData("lastError", err?.message || "Failed to load eval results")
    setData("status", "Failed to load eval results")
    log("state", "evalResults error", {
      jobId,
      error: err?.message || "unknown",
      totalMs: Date.now() - start,
    })
  }
}

export async function fetchMetrics(ctx: AppContext): Promise<void> {
  const { data } = ctx.state
  const { setData } = ctx
  const job = data.selectedJob
  if (!job) return

  const jobId = job.job_id
  const start = Date.now()
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
      log("state", "metrics stale", {
        jobId,
        totalMs: Date.now() - start,
      })
      return
    }
    setData("metrics", payload)
    setData("status", "")
    log("state", "metrics fetched", {
      jobId,
      metricKeys: payload && typeof payload === "object" ? Object.keys(payload).length : 0,
      points: Array.isArray(payload?.points) ? payload.points.length : null,
      totalMs: Date.now() - start,
    })
  } catch (err: any) {
    if (isAbortError(err)) return
    if (ctx.state.data.selectedJob?.job_id !== jobId) {
      log("state", "metrics stale", {
        jobId,
        error: err?.message || "unknown",
        totalMs: Date.now() - start,
      })
      return
    }
    setData("lastError", err?.message || "Failed to load metrics")
    setData("status", "Failed to load metrics")
    log("state", "metrics error", {
      jobId,
      error: err?.message || "unknown",
      totalMs: Date.now() - start,
    })
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
