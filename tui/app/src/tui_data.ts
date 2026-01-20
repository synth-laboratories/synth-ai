export type JobSummary = {
  job_id: string
  status: string
  job_type?: string | null
  job_source?: "prompt-learning" | "learning" | "eval" | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  best_reward?: number | null
  best_snapshot_id?: string | null
  total_tokens?: number | null
  total_cost_usd?: number | null
  error?: string | null
  metadata?: Record<string, any> | null
}

export type JobEvent = {
  seq: number
  type: string
  message?: string | null
  data?: unknown
  timestamp?: string | null
  expanded?: boolean
}

export function extractJobs(
  payload: any,
  source?: JobSummary["job_source"],
): JobSummary[] {
  const list = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.jobs)
      ? payload.jobs
      : Array.isArray(payload?.data)
        ? payload.data
        : []
  return list.map((job: any) => coerceJob(job, source))
}

export function extractEvents(
  payload: any,
): { events: JobEvent[]; nextSeq: number | null } {
  const list = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.events)
      ? payload.events
      : []
  const events = list.map((e: any, idx: number) => {
    const seqCandidate = e.seq ?? e.sequence ?? e.id
    const seqValue = Number(seqCandidate)
    return {
      seq: Number.isFinite(seqValue) ? seqValue : idx,
      type: String(e.type || e.event_type || "event"),
      message: e.message || null,
      data: e.data ?? null,
      timestamp: e.timestamp || e.created_at || null,
    }
  })
  const nextSeqRaw = payload?.next_seq
  const nextSeqValue = Number(nextSeqRaw)
  const nextSeq = Number.isFinite(nextSeqValue) ? nextSeqValue : null
  return { events, nextSeq }
}

/** Check if a job is an eval job (by source or job_type) */
export function isEvalJob(job: JobSummary | null): boolean {
  if (!job) return false
  return (
    job.job_source === "eval" ||
    job.job_type === "eval" ||
    job.job_id.startsWith("eval_")
  )
}

export function coerceJob(
  payload: any,
  source?: JobSummary["job_source"],
): JobSummary {
  const jobId = String(payload?.job_id || payload?.id || "")
  const meta = payload?.metadata
  // Extract training type from multiple possible locations
  let trainingType =
    payload?.algorithm ||
    payload?.job_type ||
    meta?.algorithm ||
    meta?.job_type ||
    meta?.prompt_initial_snapshot?.raw_config?.prompt_learning?.algorithm ||
    meta?.config?.algorithm ||
    null
  const isEval = jobId.startsWith("eval_") || trainingType === "eval"
  if (isEval && !trainingType) {
    trainingType = "eval"
  }
  const explicitSource = payload?.job_source
  const resolvedSource =
    explicitSource ||
    (isEval && source === "learning" ? "eval" : source ?? (isEval ? "eval" : null))
  return {
    job_id: jobId,
    status: String(payload?.status || "unknown"),
    // API uses 'algorithm' field, not 'job_type'
    job_type: trainingType,
    job_source: resolvedSource,
    created_at: payload?.created_at || null,
    started_at: payload?.started_at || null,
    finished_at: payload?.finished_at || null,
    best_reward: num(payload?.best_reward ?? payload?.best_score),
    best_snapshot_id: payload?.best_snapshot_id || payload?.best_snapshot?.id || null,
    total_tokens: int(payload?.total_tokens),
    total_cost_usd: num(payload?.total_cost_usd || payload?.total_cost),
    error: payload?.error || null,
    metadata: payload?.metadata || null,
  }
}

export function mergeJobs(
  primary: JobSummary[],
  secondary: JobSummary[],
): JobSummary[] {
  const byId = new Map<string, JobSummary>()
  for (const job of primary) {
    if (job.job_id) byId.set(job.job_id, job)
  }
  for (const job of secondary) {
    if (!job.job_id || byId.has(job.job_id)) continue
    byId.set(job.job_id, job)
  }
  const merged = Array.from(byId.values())
  merged.sort((a, b) => toSortTimestamp(b.created_at) - toSortTimestamp(a.created_at))
  return merged
}

export function num(value: any): number | null {
  if (value == null) return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function int(value: any): number | null {
  if (value == null) return null
  const n = parseInt(String(value), 10)
  return Number.isFinite(n) ? n : null
}

function toSortTimestamp(value?: string | null): number {
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}
