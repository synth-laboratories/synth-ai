import type { AppData } from "../types"
import type { JobSummary } from "../tui_data"
import { sortJobs } from "../tui_data"
import { log } from "../utils/log"

export type JobsIndex = {
  byId: Record<string, JobSummary>
  order: string[]
}

type JobsIndexData = Pick<AppData, "jobsById" | "jobsOrder">

const reportedIdMismatches = new Set<string>()
const reportedOrderDuplicates = new Set<string>()
const reportedMissingJobs = new Set<string>()

function cloneJobSummary(job: JobSummary): JobSummary {
  return { ...job }
}

function mergeJobSummary(existing: JobSummary, incoming: JobSummary, jobId: string): JobSummary {
  let next = existing
  if (incoming.status && incoming.status !== existing.status) {
    next = { ...next, status: incoming.status }
  }
  if (incoming.error !== undefined && incoming.error !== existing.error) {
    next = { ...next, error: incoming.error ?? null }
  }
  const stableFields: Array<keyof JobSummary> = [
    "training_type",
    "algorithm",
    "job_source",
    "model_id",
    "training_file_id",
    "eval_file_id",
    "fine_tuned_model",
    "hyperparameters",
    "cost_information",
    "linked_job_id",
    "run_id",
    "has_run",
    "created_at",
    "started_at",
    "finished_at",
    "best_reward",
    "best_train_reward",
    "best_validation_reward",
    "best_snapshot_id",
    "total_tokens",
    "total_cost_usd",
    "metadata",
  ]
  for (const field of stableFields) {
    if ((next as any)[field] == null && (incoming as any)[field] != null) {
      if (next === existing) {
        next = { ...next }
      }
      ;(next as any)[field] = (incoming as any)[field]
    }
  }
  if (next !== existing && next.job_id !== jobId) {
    next.job_id = jobId
  }
  return next
}

function buildJobsIndex(jobs: JobSummary[]): JobsIndex {
  const byId: Record<string, JobSummary> = {}
  for (const job of jobs) {
    const id = job.job_id
    if (!id) continue
    const existing = byId[id]
    if (existing) {
      byId[id] = mergeJobSummary(existing, job, id)
    } else {
      const cloned = cloneJobSummary(job)
      if (cloned.job_id !== id) cloned.job_id = id
      byId[id] = cloned
    }
  }
  const order = sortJobs(Object.values(byId)).map((job) => job.job_id)
  return { byId, order }
}

export function replaceJobsIndex(jobs: JobSummary[]): JobsIndex {
  return buildJobsIndex(jobs)
}

export function upsertJobsIndex(current: JobsIndex, jobs: JobSummary[]): JobsIndex {
  if (!jobs.length) return current
  let changed = false
  const byId: Record<string, JobSummary> = { ...current.byId }
  for (const job of jobs) {
    const id = job.job_id
    if (!id) continue
    const existing = byId[id]
    if (!existing) {
      const cloned = cloneJobSummary(job)
      if (cloned.job_id !== id) cloned.job_id = id
      byId[id] = cloned
      changed = true
      continue
    }
    const merged = mergeJobSummary(existing, job, id)
    if (merged !== existing) {
      byId[id] = merged
      changed = true
    }
  }
  if (!changed) return current
  const order = sortJobs(Object.values(byId)).map((job) => job.job_id)
  return { byId, order }
}

export function getJobsList(data: JobsIndexData): JobSummary[] {
  const list: JobSummary[] = []
  const seen = new Set<string>()
  for (const id of data.jobsOrder) {
    if (seen.has(id)) {
      if (!reportedOrderDuplicates.has(id)) {
        reportedOrderDuplicates.add(id)
        log("state", "jobs order duplicate", { id })
      }
      continue
    }
    seen.add(id)
    const job = data.jobsById[id]
    if (!job) {
      if (!reportedMissingJobs.has(id)) {
        reportedMissingJobs.add(id)
        log("state", "jobs index missing", { orderId: id })
      }
      continue
    }
    if (job.job_id !== id) {
      const key = `${id}:${job.job_id ?? "missing"}`
      if (!reportedIdMismatches.has(key)) {
        reportedIdMismatches.add(key)
        log("state", "jobs index id mismatch", { orderId: id, jobId: job.job_id ?? null })
      }
      list.push({ ...job, job_id: id })
    } else {
      list.push(job)
    }
  }
  return list
}

export function getJobById(data: Pick<AppData, "jobsById">, jobId: string): JobSummary | null {
  if (!jobId) return null
  const job = data.jobsById[jobId]
  if (!job) return null
  if (job.job_id !== jobId) {
    const key = `${jobId}:${job.job_id ?? "missing"}`
    if (!reportedIdMismatches.has(key)) {
      reportedIdMismatches.add(key)
      log("state", "jobs index id mismatch", { orderId: jobId, jobId: job.job_id ?? null })
    }
    return { ...job, job_id: jobId }
  }
  return job
}

export function getJobsIndex(data: JobsIndexData): JobsIndex {
  return { byId: data.jobsById, order: data.jobsOrder }
}

export function cloneJobsIndex(index: JobsIndex): JobsIndex {
  return { byId: { ...index.byId }, order: [...index.order] }
}

export function cloneJob(job: JobSummary): JobSummary {
  return cloneJobSummary(job)
}
