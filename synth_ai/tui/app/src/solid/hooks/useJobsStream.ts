import { createEffect, onCleanup, type Accessor } from "solid-js"

import type { AppContext } from "../../context"
import { refreshJobs, recordJobsCache } from "../../api/jobs"
import { connectJobsStream, type JobStreamEvent, type JobStreamConnection } from "../../api/jobs-stream"
import type { JobSummary } from "../../tui_data"
import { sortJobs } from "../../tui_data"
import { log, logError } from "../../utils/log"
import { registerCleanup, unregisterCleanup } from "../../lifecycle"
import { isTerminalJobStatus } from "../../utils/job"

type UseJobsStreamOptions = {
  ctx: AppContext
  enabled?: Accessor<boolean>
}

function inferJobSource(
  jobId: string,
  jobType?: string,
): JobSummary["job_source"] | null {
  if (jobId.startsWith("eval_")) return "eval"
  if (jobId.startsWith("pl_")) return "prompt-learning"
  if (jobId.startsWith("learning_")) return "learning"
  if (jobType) {
    const normalized = jobType.toLowerCase()
    if (normalized.includes("eval")) return "eval"
    if (normalized.includes("prompt")) return "prompt-learning"
    if (normalized.includes("learning")) return "learning"
  }
  return null
}

function buildJobPatch(event: JobStreamEvent): Partial<JobSummary> {
  const patch: Partial<JobSummary> = {}
  if (event.status) patch.status = event.status
  if (event.job_type) patch.training_type = event.job_type
  if (event.algorithm) patch.algorithm = event.algorithm
  if (event.model_id) patch.model_id = event.model_id
  if (event.created_at) patch.created_at = event.created_at
  if (event.started_at) patch.started_at = event.started_at
  if (event.finished_at) patch.finished_at = event.finished_at
  if (event.error) patch.error = event.error
  const source = inferJobSource(event.job_id, event.job_type)
  if (source) patch.job_source = source
  return patch
}

function createJobFromEvent(event: JobStreamEvent, patch: Partial<JobSummary>): JobSummary {
  const fallbackCreatedAt =
    !event.created_at && typeof event.ts === "number"
      ? new Date(event.ts).toISOString()
      : null
  return {
    job_id: event.job_id,
    status: event.status || "unknown",
    training_type: patch.training_type ?? null,
    algorithm: patch.algorithm ?? null,
    job_source: patch.job_source ?? inferJobSource(event.job_id, event.job_type),
    model_id: patch.model_id ?? null,
    training_file_id: null,
    eval_file_id: null,
    fine_tuned_model: null,
    hyperparameters: null,
    cost_information: null,
    linked_job_id: null,
    run_id: null,
    has_run: null,
    created_at: patch.created_at ?? fallbackCreatedAt,
    started_at: patch.started_at ?? null,
    finished_at: patch.finished_at ?? null,
    best_reward: null,
    best_train_reward: null,
    best_validation_reward: null,
    best_snapshot_id: null,
    total_tokens: null,
    total_cost_usd: null,
    error: patch.error ?? null,
    metadata: null,
  }
}

function mergeJobUpdate(job: JobSummary, patch: Partial<JobSummary>): JobSummary {
  return {
    ...job,
    ...patch,
    training_type: job.training_type ?? patch.training_type ?? null,
    job_source: job.job_source ?? patch.job_source ?? null,
  }
}

export function useJobsStream(options: UseJobsStreamOptions): void {
  const cleanupName = "jobs-stream"
  let connection: JobStreamConnection | null = null
  let lastSeq = 0
  let refreshTimer: ReturnType<typeof setTimeout> | null = null

  const cleanup = (reason?: string) => {
    if (refreshTimer) {
      clearTimeout(refreshTimer)
      refreshTimer = null
    }
    if (connection) {
      log("state", `jobs-stream disconnect${reason ? ` (${reason})` : ""}`)
      connection.disconnect()
      connection = null
    }
  }

  const scheduleRefresh = () => {
    if (refreshTimer) return
    refreshTimer = setTimeout(() => {
      refreshTimer = null
      void refreshJobs(options.ctx)
    }, 500)
  }

  const onEvent = (event: JobStreamEvent) => {
    if (!event.job_id) return
    if (typeof event.seq === "number") {
      if (event.seq <= lastSeq) return
      lastSeq = event.seq
    }
    const patch = buildJobPatch(event)
    const existing = options.ctx.state.data.jobs.find((job) => job.job_id === event.job_id)
    const cacheCandidate = existing
      ? mergeJobUpdate(existing, patch)
      : createJobFromEvent(event, patch)
    options.ctx.setData("jobs", (prev) => {
      const idx = prev.findIndex((job) => job.job_id === event.job_id)
      if (idx >= 0) {
        const updated = mergeJobUpdate(prev[idx], patch)
        const next = [...prev]
        next[idx] = updated
        if (event.status && prev[idx].status !== event.status) {
          log("state", "jobs-stream status", {
            jobId: event.job_id,
            from: prev[idx].status,
            to: event.status,
            seq: event.seq,
          })
        }
        return sortJobs(next)
      }
      const created = createJobFromEvent(event, patch)
      log("state", "jobs-stream new job", {
        jobId: event.job_id,
        status: created.status,
        seq: event.seq,
      })
      return sortJobs([...prev, created])
    })

    if (options.ctx.state.data.selectedJob?.job_id === event.job_id) {
      options.ctx.setData("selectedJob", (current) => {
        if (!current) return current
        return mergeJobUpdate(current, patch)
      })
    }

    if (isTerminalJobStatus(cacheCandidate.status)) {
      recordJobsCache(options.ctx, [cacheCandidate])
    }
  }

  const onError = (err: Error) => {
    logError("jobs-stream error", err)
    scheduleRefresh()
  }

  createEffect(() => {
    const enabled = options.enabled ? options.enabled() : true
    if (!enabled) {
      cleanup("disabled")
      lastSeq = 0
      return
    }
    cleanup("reconnect")
    log("state", "jobs-stream connect")
    connection = connectJobsStream(onEvent, onError, () => lastSeq)
    registerCleanup(cleanupName, () => cleanup("shutdown"))
  })

  onCleanup(() => {
    cleanup("dispose")
    unregisterCleanup(cleanupName)
  })
}
