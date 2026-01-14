import { promises as fs } from "node:fs"

import type { Mode } from "../types"
import type { JobSummary } from "../tui_data"
import { tuiCacheDir, tuiJobsCachePath } from "../paths"
import { isTerminalJobStatus } from "../utils/job"
import { logError } from "../utils/log"

const CACHE_VERSION = 1
const MAX_CACHE_JOBS = 1000
const WRITE_DEBOUNCE_MS = 1000

export type CachedJobSummary = Pick<
  JobSummary,
  | "job_id"
  | "status"
  | "training_type"
  | "job_source"
  | "algorithm"
  | "model_id"
  | "created_at"
  | "started_at"
  | "finished_at"
  | "error"
  | "best_snapshot_id"
  | "best_reward"
  | "best_train_reward"
  | "best_validation_reward"
>

type JobsCacheEntry = {
  updatedAt: string
  jobs: CachedJobSummary[]
}

type JobsCacheFile = {
  version: number
  updatedAt: string
  entries: Record<string, JobsCacheEntry>
}

const pendingWrites = new Map<string, CachedJobSummary[]>()
let writeTimer: ReturnType<typeof setTimeout> | null = null

export function getJobsCacheKey(orgId: string, mode: Mode): string {
  return `${mode}:${orgId}`
}

function getJobTimestamp(job: Pick<JobSummary, "created_at" | "started_at" | "finished_at">): number {
  const value = job.created_at || job.started_at || job.finished_at
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function inferJobSource(jobId: string): JobSummary["job_source"] | null {
  if (jobId.startsWith("eval_")) return "eval"
  if (jobId.startsWith("pl_")) return "prompt-learning"
  if (jobId.startsWith("learning_")) return "learning"
  return null
}

function toCachedJob(job: JobSummary): CachedJobSummary {
  const resolvedSource = job.job_source ?? inferJobSource(job.job_id)
  return {
    job_id: job.job_id,
    status: job.status,
    training_type: job.training_type ?? null,
    job_source: resolvedSource ?? null,
    algorithm: job.algorithm ?? null,
    model_id: job.model_id ?? null,
    created_at: job.created_at ?? null,
    started_at: job.started_at ?? null,
    finished_at: job.finished_at ?? null,
    error: job.error ?? null,
    best_snapshot_id: job.best_snapshot_id ?? null,
    best_reward: job.best_reward ?? null,
    best_train_reward: job.best_train_reward ?? null,
    best_validation_reward: job.best_validation_reward ?? null,
  }
}

function trimCache(jobs: CachedJobSummary[]): CachedJobSummary[] {
  const sorted = [...jobs].sort((a, b) => getJobTimestamp(b) - getJobTimestamp(a))
  if (sorted.length <= MAX_CACHE_JOBS) return sorted
  return sorted.slice(0, MAX_CACHE_JOBS)
}

function normalizeCacheFile(payload: unknown): JobsCacheFile {
  if (!payload || typeof payload !== "object") {
    return { version: CACHE_VERSION, updatedAt: new Date().toISOString(), entries: {} }
  }
  const data = payload as Partial<JobsCacheFile>
  return {
    version: typeof data.version === "number" ? data.version : CACHE_VERSION,
    updatedAt: typeof data.updatedAt === "string" ? data.updatedAt : new Date().toISOString(),
    entries: data.entries && typeof data.entries === "object" ? data.entries : {},
  }
}

async function readCacheFile(): Promise<JobsCacheFile> {
  try {
    const content = await fs.readFile(tuiJobsCachePath, "utf8")
    return normalizeCacheFile(JSON.parse(content))
  } catch (err: any) {
    if (err?.code === "ENOENT") {
      return { version: CACHE_VERSION, updatedAt: new Date().toISOString(), entries: {} }
    }
    logError("jobs-cache read failed", err)
    return { version: CACHE_VERSION, updatedAt: new Date().toISOString(), entries: {} }
  }
}

async function writeCacheFile(file: JobsCacheFile): Promise<void> {
  try {
    await fs.mkdir(tuiCacheDir, { recursive: true })
    await fs.writeFile(tuiJobsCachePath, `${JSON.stringify(file, null, 2)}\n`, "utf8")
  } catch (err) {
    logError("jobs-cache write failed", err)
  }
}

export async function loadJobsCache(key: string): Promise<CachedJobSummary[]> {
  const file = await readCacheFile()
  const entry = file.entries[key]
  if (!entry || !Array.isArray(entry.jobs)) return []
  const filtered = entry.jobs.filter((job) => job?.job_id && job?.status)
  const normalized = filtered.map((job) => toCachedJob(job as JobSummary))
  return trimCache(normalized.filter((job) => isTerminalJobStatus(job.status)))
}

function mergeCachedJob(previous: CachedJobSummary, next: CachedJobSummary): CachedJobSummary {
  const merged: CachedJobSummary = { ...previous, ...next }
  const fields: Array<keyof CachedJobSummary> = [
    "training_type",
    "job_source",
    "algorithm",
    "model_id",
    "created_at",
    "started_at",
    "finished_at",
    "error",
    "best_snapshot_id",
    "best_reward",
    "best_train_reward",
    "best_validation_reward",
  ]
  for (const field of fields) {
    if (next[field] == null && previous[field] != null) {
      ;(merged as Record<string, unknown>)[field] = previous[field]
    }
  }
  if (!next.status && previous.status) {
    merged.status = previous.status
  }
  return merged
}

export function mergeJobsCache(
  existing: CachedJobSummary[],
  incoming: JobSummary[],
): CachedJobSummary[] {
  if (!incoming.length) return existing
  const byId = new Map<string, CachedJobSummary>()
  for (const job of existing) {
    if (job.job_id) byId.set(job.job_id, job)
  }
  for (const job of incoming) {
    if (!job.job_id || !isTerminalJobStatus(job.status)) continue
    const cached = toCachedJob(job)
    const prev = byId.get(job.job_id)
    if (!prev) {
      byId.set(job.job_id, cached)
      continue
    }
    const prevTs = getJobTimestamp(prev)
    const nextTs = getJobTimestamp(cached)
    const primary = nextTs >= prevTs ? cached : prev
    const secondary = nextTs >= prevTs ? prev : cached
    byId.set(job.job_id, mergeCachedJob(secondary, primary))
  }
  return trimCache(Array.from(byId.values()))
}

async function flushJobsCacheWrites(): Promise<void> {
  if (!pendingWrites.size) return
  const entries = new Map(pendingWrites)
  pendingWrites.clear()
  writeTimer = null

  const file = await readCacheFile()
  const now = new Date().toISOString()
  file.updatedAt = now
  for (const [key, jobs] of entries) {
    file.entries[key] = { updatedAt: now, jobs: trimCache(jobs) }
  }
  await writeCacheFile(file)
}

export function scheduleJobsCacheWrite(key: string, jobs: CachedJobSummary[]): void {
  pendingWrites.set(key, trimCache(jobs))
  if (writeTimer) return
  writeTimer = setTimeout(() => {
    void flushJobsCacheWrites()
  }, WRITE_DEBOUNCE_MS)
}
