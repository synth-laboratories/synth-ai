import { createEffect, onCleanup, type Accessor } from "solid-js"

import type { AppContext } from "../../context"
import { refreshJobs, recordJobsCache } from "../../api/jobs"
import { connectJobsStream, type JobStreamEvent, type JobStreamConnection } from "../../api/jobs-stream"
import { refreshHealth } from "../../api/identity"
import type { JobSummary } from "../../tui_data"
import { log, logError } from "../../utils/log"
import { registerCleanup, unregisterCleanup } from "../../lifecycle"
import { isTerminalJobStatus } from "../../utils/job"
import { isPollScheduledSoon, pollingState, setSseConnected, shouldPoll } from "../../state/polling"
import { getJobById, getJobsIndex, upsertJobsIndex } from "../../state/jobs-index"

type UseJobsStreamOptions = {
  ctx: AppContext
  enabled?: Accessor<boolean>
}

const JOBS_STREAM_BACKFILL_MS = 5000

function normalizeEventTs(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null
  return value < 1_000_000_000_000 ? value * 1000 : value
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

function buildStatusPatch(event: JobStreamEvent): Partial<JobSummary> {
  const patch: Partial<JobSummary> = {}
  if (event.status) patch.status = event.status
  if (event.error) patch.error = event.error
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

function countJobIdOccurrences(ids: string[], jobId: string): number {
  let count = 0
  for (const id of ids) {
    if (id === jobId) count += 1
  }
  return count
}

type DuplicateJobId = {
  id: string
  count: number
  indices: number[]
}

function findDuplicateJobIds(ids: string[]): DuplicateJobId[] {
  const counts = new Map<string, DuplicateJobId>()
  ids.forEach((id, idx) => {
    if (!id) {
      return
    }
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

export function useJobsStream(options: UseJobsStreamOptions): void {
  const cleanupName = "jobs-stream"
  let connection: JobStreamConnection | null = null
  let lastSeq = 0
  let refreshTimer: ReturnType<typeof setTimeout> | null = null
  let streamStartMs = 0

  const cleanup = (reason?: string) => {
    if (refreshTimer) {
      clearTimeout(refreshTimer)
      refreshTimer = null
    }
    setSseConnected("jobs", false)
    streamStartMs = 0
    if (connection) {
      log("state", `jobs-stream disconnect${reason ? ` (${reason})` : ""}`)
      connection.disconnect()
      connection = null
    }
  }

  const scheduleRefresh = () => {
    if (!shouldPoll("jobs")) return
    if (pollingState.jobsInFlight) return
    if (isPollScheduledSoon("jobs", 1000)) return
    if (refreshTimer) return
    refreshTimer = setTimeout(() => {
      refreshTimer = null
      if (!shouldPoll("jobs")) return
      if (pollingState.jobsInFlight) return
      if (isPollScheduledSoon("jobs", 1000)) return
      pollingState.jobsInFlight = true
      void refreshJobs(options.ctx).then((result) => {
        if (!result.ok) {
          void refreshHealth(options.ctx)
        }
      }).finally(() => {
        pollingState.jobsInFlight = false
      })
    }, 500)
  }

  const onEvent = (event: JobStreamEvent) => {
    if (!event.job_id) return
    if (typeof event.seq === "number") {
      if (event.seq <= lastSeq) return
      lastSeq = event.seq
    }
    const eventTs = normalizeEventTs(event.ts)
    const isBackfill =
      streamStartMs > 0 &&
      eventTs != null &&
      eventTs < streamStartMs - JOBS_STREAM_BACKFILL_MS

    const existing = getJobById(options.ctx.state.data, event.job_id)
    if (existing) {
      const existingCount = countJobIdOccurrences(options.ctx.state.data.jobsOrder, event.job_id)
      if (existingCount > 1) {
        log("state", "jobs-stream duplicate existing", {
          jobId: event.job_id,
          count: existingCount,
          eventType: event.type,
          eventStatus: event.status,
          seq: event.seq,
        })
      }
      if (isTerminalJobStatus(existing.status)) {
        log("state", "jobs-stream skip terminal", {
          jobId: event.job_id,
          status: existing.status,
          eventStatus: event.status,
          eventType: event.type,
          seq: event.seq,
        })
        return
      }
      const patch = buildStatusPatch(event)
      if (!patch.status && !patch.error) {
        log("state", "jobs-stream skip noop", {
          jobId: event.job_id,
          eventType: event.type,
          eventStatus: event.status,
          seq: event.seq,
        })
        return
      }
      if (
        patch.status &&
        patch.status === existing.status &&
        (patch.error === undefined || patch.error === existing.error)
      ) {
        log("state", "jobs-stream skip unchanged", {
          jobId: event.job_id,
          status: existing.status,
          eventStatus: event.status,
          seq: event.seq,
        })
        return
      }
      const prevIndex = getJobsIndex(options.ctx.state.data)
      const updated = { ...existing, ...patch }
      const nextIndex = upsertJobsIndex(prevIndex, [updated])
      if (patch.status && existing.status !== patch.status) {
        log("state", "jobs-stream status", {
          jobId: event.job_id,
          from: existing.status,
          to: patch.status,
          seq: event.seq,
        })
      }
      if (nextIndex !== prevIndex) {
        const prevIds = prevIndex.order
        const nextIds = nextIndex.order
        const { lengthChanged, changedCount } = countOrderChanges(prevIds, nextIds)
        const prevDuplicates = findDuplicateJobIds(prevIds)
        const nextDuplicates = findDuplicateJobIds(nextIds)
        const duplicatesChanged = buildDuplicateKey(prevDuplicates) !== buildDuplicateKey(nextDuplicates)
        const orderChanged = lengthChanged || changedCount > 0
        if (orderChanged || duplicatesChanged) {
          log("state", "jobs-stream apply update", {
            jobId: event.job_id,
            eventType: event.type,
            eventStatus: event.status,
            seq: event.seq,
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
        }
        options.ctx.setData({ jobsById: nextIndex.byId, jobsOrder: nextIndex.order })
      }

      if (options.ctx.state.data.selectedJob?.job_id === event.job_id) {
        options.ctx.setData("selectedJob", (current) => {
          if (!current) return current
          return { ...current, ...patch }
        })
      }

      const cacheCandidate = { ...existing, ...patch }
      if (isTerminalJobStatus(cacheCandidate.status)) {
        recordJobsCache(options.ctx, [cacheCandidate])
      }
      return
    }

    if (isBackfill) {
      log("state", "jobs-stream skip backfill", {
        jobId: event.job_id,
        eventType: event.type,
        eventStatus: event.status,
        eventTs,
        streamStartMs,
        seq: event.seq,
      })
      return
    }

    if (event.type !== "job.created") {
      log("state", "jobs-stream skip new job", {
        jobId: event.job_id,
        eventType: event.type,
        eventStatus: event.status,
        seq: event.seq,
      })
      return
    }

    const patch = buildJobPatch(event)
    const created = createJobFromEvent(event, patch)
    log("state", "jobs-stream new job", {
      jobId: event.job_id,
      status: created.status,
      seq: event.seq,
    })
    const prevIndex = getJobsIndex(options.ctx.state.data)
    const prevIds = prevIndex.order
    const count = countJobIdOccurrences(prevIds, event.job_id)
    if (count > 0) {
      log("state", "jobs-stream duplicate insert", {
        jobId: event.job_id,
        count,
        eventType: event.type,
        eventStatus: event.status,
        seq: event.seq,
      })
    }
    const nextIndex = upsertJobsIndex(prevIndex, [created])
    if (nextIndex !== prevIndex) {
      const nextIds = nextIndex.order
      const { lengthChanged, changedCount } = countOrderChanges(prevIds, nextIds)
      const prevDuplicates = findDuplicateJobIds(prevIds)
      const nextDuplicates = findDuplicateJobIds(nextIds)
      const duplicatesChanged = buildDuplicateKey(prevDuplicates) !== buildDuplicateKey(nextDuplicates)
      const orderChanged = lengthChanged || changedCount > 0
      if (orderChanged || duplicatesChanged) {
        log("state", "jobs-stream apply insert", {
          jobId: event.job_id,
          eventType: event.type,
          eventStatus: event.status,
          seq: event.seq,
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
      }
      options.ctx.setData({ jobsById: nextIndex.byId, jobsOrder: nextIndex.order })
    }

    if (isTerminalJobStatus(created.status)) {
      recordJobsCache(options.ctx, [created])
    }
  }

  const onError = (err: Error) => {
    setSseConnected("jobs", false)
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
    connection = connectJobsStream(onEvent, onError, () => lastSeq, {
      onOpen: () => {
        streamStartMs = Date.now()
        setSseConnected("jobs", true)
      },
    })
    registerCleanup(cleanupName, () => cleanup("shutdown"))
  })

  onCleanup(() => {
    cleanup("dispose")
    unregisterCleanup(cleanupName)
  })
}
