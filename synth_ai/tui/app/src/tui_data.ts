export type JobSummary = {
  job_id: string
  status: string
  training_type?: string | null
  algorithm?: string | null
  job_source?: "prompt-learning" | "learning" | "eval" | null
  model_id?: string | null
  training_file_id?: string | null
  eval_file_id?: string | null
  fine_tuned_model?: string | null
  hyperparameters?: Record<string, any> | null
  cost_information?: Record<string, any> | null
  linked_job_id?: string | null
  run_id?: string | null
  has_run?: boolean | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  best_reward?: number | null
  best_train_reward?: number | null
  best_validation_reward?: number | null
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

/** Check if a job is an eval job (by source or training_type) */
export function isEvalJob(job: JobSummary | null): boolean {
  if (!job) return false
  return (
    job.job_source === "eval" ||
    job.training_type === "eval" ||
    job.job_id.startsWith("eval_")
  )
}

/**
 * Normalize job type to surface hidden distinctions from metadata.
 * Converts generic types like "graph_evolve" into specific types like "verifier_evolve".
 */
export function normalizeJobType(job: JobSummary | null): string {
  if (!job) return "unknown"

  const trainingType = job.training_type
  const metadata = job.metadata

  // graph_evolve: distinguish policy vs verifier optimization
  if (trainingType === "graph_evolve") {
    const graphType = metadata?.graph_type
    if (graphType === "verifier") return "verifier_evolve"
    if (graphType === "policy") return "policy_evolve"
    if (graphType === "rlm") return "rlm_evolve"
    return "graph_evolve" // fallback if graph_type not set
  }

  // prompt_learning: surface the algorithm (gepa vs mipro)
  if (trainingType === "prompt_learning") {
    const algorithm = metadata?.algorithm
    if (algorithm === "gepa") return "prompt_gepa"
    if (algorithm === "mipro") return "prompt_mipro"
    return "prompt_learning" // fallback
  }

  // eval: surface eval mode if not default task_app
  if (trainingType === "eval" || job.job_source === "eval" || job.job_id.startsWith("eval_")) {
    const evalMode = metadata?.eval_mode
    if (evalMode === "verifier") return "eval_verifier"
    if (evalMode === "graph") return "eval_graph"
    return "eval" // task_app mode or default
  }

  // Return as-is for other types (sft_offline, sft_online, rl_online, etc.)
  return trainingType || job.job_source || "unknown"
}

export function coerceJob(
  payload: any,
  source?: JobSummary["job_source"],
): JobSummary {
  const jobId = String(payload?.job_id || payload?.id || "")
  const meta = payload?.metadata
  const hyperparameters =
    payload?.hyperparameters && typeof payload.hyperparameters === "object" && !Array.isArray(payload.hyperparameters)
      ? payload.hyperparameters
      : null
  const costInformation =
    payload?.cost_information && typeof payload.cost_information === "object" && !Array.isArray(payload.cost_information)
      ? payload.cost_information
      : null
  // Extract training type from multiple possible locations
  let trainingType =
    payload?.algorithm ||
    payload?.training_type ||
    meta?.algorithm ||
    meta?.training_type ||
    meta?.prompt_initial_snapshot?.raw_config?.prompt_learning?.algorithm ||
    meta?.config?.algorithm ||
    null
  const algorithm =
    payload?.algorithm ||
    meta?.algorithm ||
    meta?.config?.algorithm ||
    meta?.prompt_initial_snapshot?.raw_config?.prompt_learning?.algorithm ||
    null
  const isEval = jobId.startsWith("eval_") || trainingType === "eval"
  if (isEval && !trainingType) {
    trainingType = "eval"
  }
  const explicitSource = payload?.job_source
  const resolvedSource =
    explicitSource ||
    (isEval && source === "learning" ? "eval" : source ?? (isEval ? "eval" : null))
  const bestReward = num(
    payload?.best_reward ??
      payload?.best_score ??
      meta?.prompt_best_score ??
      meta?.prompt_best_reward ??
      meta?.best_reward ??
      meta?.best_score,
  )
  const bestTrainReward = num(
    payload?.best_train_reward ??
      payload?.best_train_score ??
      meta?.prompt_best_train_score ??
      meta?.prompt_best_train_reward ??
      meta?.best_train_reward ??
      meta?.best_train_score,
  )
  const bestValidationReward = num(
    payload?.best_validation_reward ??
      payload?.best_validation_score ??
      meta?.prompt_best_validation_score ??
      meta?.prompt_best_validation_reward ??
      meta?.best_validation_reward ??
      meta?.best_validation_score,
  )
  return {
    job_id: jobId,
    status: String(payload?.status || "unknown"),
    // API uses 'algorithm' field, not 'training_type'
    training_type: trainingType,
    algorithm,
    job_source: resolvedSource,
    model_id: payload?.model_id ?? meta?.model_id ?? null,
    training_file_id: payload?.training_file_id ?? null,
    eval_file_id: payload?.eval_file_id ?? null,
    fine_tuned_model: payload?.fine_tuned_model ?? meta?.fine_tuned_model ?? null,
    hyperparameters,
    cost_information: costInformation,
    linked_job_id: payload?.linked_job_id ?? null,
    run_id: payload?.run_id ?? null,
    has_run: typeof payload?.has_run === "boolean" ? payload.has_run : null,
    created_at: payload?.created_at || null,
    started_at: payload?.started_at || null,
    finished_at: payload?.finished_at || null,
    best_reward: bestReward,
    best_train_reward: bestTrainReward,
    best_validation_reward: bestValidationReward,
    best_snapshot_id:
      payload?.best_snapshot_id ||
      payload?.best_snapshot?.id ||
      meta?.prompt_best_snapshot_id ||
      meta?.best_snapshot_id ||
      null,
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
  merged.sort(compareJobsByDate)
  return merged
}

export function sortJobs(jobs: JobSummary[]): JobSummary[] {
  const next = [...jobs]
  next.sort(compareJobsByDate)
  return next
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

function getJobSortKey(job: JobSummary): { ts: number; id: string } {
  const created = toSortTimestamp(job.created_at)
  const started = toSortTimestamp(job.started_at)
  const finished = toSortTimestamp(job.finished_at)
  const ts = created || started || finished || 0
  return { ts, id: job.job_id || "" }
}

function compareJobsByDate(a: JobSummary, b: JobSummary): number {
  const aKey = getJobSortKey(a)
  const bKey = getJobSortKey(b)
  if (bKey.ts !== aKey.ts) return bKey.ts - aKey.ts
  return aKey.id.localeCompare(bKey.id)
}

function toSortTimestamp(value?: string | null): number {
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}
