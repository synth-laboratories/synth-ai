/**
 * SSE client for real-time job details updates from /api/prompt-learning/online/jobs/{job_id}/events/stream
 * Works for ALL job types: eval, learning, prompt-learning
 * 
 * Ported from feat/job-details branch.
 */

import { buildApiUrl, getAuthHeaders } from "../../api/client"
import { connectSse } from "../../utils/sse"

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
  sinceSeq: number = 0,
): JobDetailsStreamConnection {
  // Use prompt-learning SSE endpoint (works for all jobs in learning_jobs table)
  const url = buildApiUrl(`/prompt-learning/online/jobs/${jobId}/events/stream?since_seq=${sinceSeq}`)
  let headers: HeadersInit
  try {
    headers = {
      ...getAuthHeaders(),
      Accept: "text/event-stream",
    }
  } catch (err) {
    onError?.(err instanceof Error ? err : new Error(String(err)))
    return {
      disconnect: () => {},
      jobId,
    }
  }

  const connection = connectSse(url, {
    headers,
    includeScope: false,
    onMessage: (message) => {
      if (!message.data) return
      try {
        const data = JSON.parse(message.data) as JobDetailsStreamEvent
        onEvent(data)
      } catch {
        // Ignore parse errors
      }
    },
    onError,
  })

  return {
    disconnect: () => {
      connection.disconnect()
    },
    jobId,
  }
}

