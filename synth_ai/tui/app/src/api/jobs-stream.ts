/**
 * SSE client for real-time job updates from /api/jobs/stream
 */

import { buildApiUrl, getAuthHeaders } from "./client"
import { connectSse } from "../utils/sse"

export interface JobStreamEvent {
	org_id: string
	job_id: string
	job_type: string
	status: string
	type: string // job.created, job.started, job.completed, job.failed
	seq: number
	ts: number
	message?: string
	model_id?: string
	algorithm?: string
	backend?: string
	error?: string
	created_at?: string
	started_at?: string
	finished_at?: string
}

export type JobStreamHandler = (event: JobStreamEvent) => void
export type JobStreamErrorHandler = (err: Error) => void

export interface JobStreamConnection {
	disconnect: () => void
}

/**
 * Connect to the jobs SSE stream.
 * Returns a connection object with a disconnect() method.
 */
export function connectJobsStream(
	onEvent: JobStreamHandler,
	onError?: JobStreamErrorHandler,
	sinceSeq: number = 0,
): JobStreamConnection {
	const url = buildApiUrl(`/jobs/stream?since_seq=${sinceSeq}`)
	let headers: HeadersInit
	try {
		headers = {
			...getAuthHeaders(),
			Accept: "text/event-stream",
		}
	} catch (err) {
		onError?.(err instanceof Error ? err : new Error(String(err)))
		return { disconnect: () => {} }
	}

	const connection = connectSse(url, {
		headers,
		includeScope: false,
		onMessage: (message) => {
			if (!message.data) return
			try {
				onEvent(JSON.parse(message.data) as JobStreamEvent)
			} catch {
				// Ignore parse errors
			}
		},
		onError,
	})

	return {
		disconnect: () => connection.disconnect(),
	}
}
