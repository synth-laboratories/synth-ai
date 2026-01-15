/**
 * SSE client for real-time job updates from /api/jobs/stream
 */

import { connectApiJsonStream } from "./stream"

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
	sinceSeq: number | (() => number) = 0,
	options: { signal?: AbortSignal } = {},
): JobStreamConnection {
	const getSinceSeq = typeof sinceSeq === "function" ? sinceSeq : () => sinceSeq
	const getPath = () => `/jobs/stream?since_seq=${getSinceSeq()}`
	const connection = connectApiJsonStream<JobStreamEvent>({
		path: getPath,
		signal: options.signal,
		includeScope: false,
		label: "jobs-stream",
		onEvent,
		onError,
	})

	return {
		disconnect: () => connection.disconnect(),
	}
}
