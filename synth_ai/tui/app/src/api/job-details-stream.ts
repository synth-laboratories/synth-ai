/**
 * SSE client for real-time job details updates from /api/prompt-learning/online/jobs/{job_id}/events/stream
 * Works for ALL job types: eval, learning, prompt-learning
 * 
 * Ported from feat/job-details branch.
 */

import { connectApiJsonStream } from "./stream"

export interface JobDetailsStreamEvent {
  job_id: string
  seq: number
  ts: number
  type: string // e.g., eval.job.started, learning.iteration.completed, prompt.learning.progress
  level: string
  message: string
  run_id?: string | null
  data: Record<string, unknown> // Generic data payload - varies by job type
}

export type JobDetailsStreamHandler = (event: JobDetailsStreamEvent) => void
export type JobDetailsStreamErrorHandler = (err: Error) => void

export interface JobDetailsStreamConnection {
  disconnect: () => void
  jobId: string
}

/**
 * Connect to the job details SSE stream.
 * Works for any job type (eval, learning, prompt-learning).
 * Returns a connection object with a disconnect() method.
 */
export function connectJobDetailsStream(
  jobId: string,
  onEvent: JobDetailsStreamHandler,
  onError?: JobDetailsStreamErrorHandler,
  sinceSeq: number | (() => number) = 0,
  options: { signal?: AbortSignal; onOpen?: () => void } = {},
): JobDetailsStreamConnection {
  const getSinceSeq = typeof sinceSeq === "function" ? sinceSeq : () => sinceSeq
  const getPath = () =>
    `/prompt-learning/online/jobs/${jobId}/events/stream?since_seq=${getSinceSeq()}`
  const connection = connectApiJsonStream<JobDetailsStreamEvent>({
    path: getPath,
    includeScope: false,
    label: `job-details:${jobId}`,
    onEvent,
    onError,
    onOpen: options.onOpen,
    signal: options.signal,
  })

  return {
    disconnect: () => {
      connection.disconnect()
    },
    jobId,
  }
}
