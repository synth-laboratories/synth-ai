/**
 * SSE client for real-time eval job updates from /api/eval/jobs/{job_id}/stream
 */

export interface EvalStreamEvent {
	job_id: string
	seq: number
	ts: number
	type: string // eval.job.started, eval.seed.completed, eval.job.progress, etc.
	level: string
	message: string
	run_id?: string | null
	data: {
		// eval.job.progress
		completed?: number
		total?: number
		// eval.seed.completed / eval.results.updated
		seed?: number
		trial_id?: string
		correlation_id?: string
		score?: number | null
		outcome_reward?: number | null
		events_score?: number | null
		verifier_score?: number | null
		latency_ms?: number | null
		tokens?: number | null
		cost_usd?: number | null
		error?: string | null
		trace_id?: string | null
		// eval.job.completed
		mean_reward?: number | null
		failed?: number
		// eval.job.started
		seed_count?: number
		// generic
		[key: string]: unknown
	}
}

export type EvalStreamHandler = (event: EvalStreamEvent) => void
export type EvalStreamErrorHandler = (err: Error) => void

export interface EvalStreamConnection {
	disconnect: () => void
	jobId: string
}

/**
 * Connect to the eval job SSE stream.
 * Returns a connection object with a disconnect() method.
 */
export function connectEvalStream(
	jobId: string,
	onEvent: EvalStreamHandler,
	onError?: EvalStreamErrorHandler,
	sinceSeq: number = 0,
): EvalStreamConnection {
	let aborted = false
	const controller = new AbortController()

	const url = `${process.env.SYNTH_BACKEND_URL}/api/eval/jobs/${jobId}/stream?since_seq=${sinceSeq}`
	const apiKey = process.env.SYNTH_API_KEY || ""

	// Start streaming in the background
	void (async () => {
		try {
			const res = await fetch(url, {
				headers: {
					Authorization: `Bearer ${apiKey}`,
					Accept: "text/event-stream",
				},
				signal: controller.signal,
			})

			if (!res.ok) {
				const body = await res.text().catch(() => "")
				throw new Error(`Eval SSE stream failed: HTTP ${res.status} ${res.statusText} - ${body.slice(0, 100)}`)
			}

			if (!res.body) {
				throw new Error("Eval SSE stream: no response body")
			}

			// Parse SSE stream
			const reader = res.body.getReader()
			const decoder = new TextDecoder()
			let buffer = ""
			let currentEvent: { type?: string; data?: string; id?: string } = {}

			while (!aborted) {
				const { done, value } = await reader.read()
				if (done) break

				buffer += decoder.decode(value, { stream: true })

				// Process complete lines
				const lines = buffer.split("\n")
				buffer = lines.pop() ?? "" // Keep incomplete line in buffer

				for (const line of lines) {
					if (line.startsWith(":")) {
						// Comment (keepalive), ignore
						continue
					}

					if (line === "") {
						// Empty line = dispatch event
						if (currentEvent.data) {
							try {
								const data = JSON.parse(currentEvent.data) as EvalStreamEvent
								onEvent(data)
							} catch {
								// Ignore parse errors
							}
						}
						currentEvent = {}
						continue
					}

					// Parse SSE field
					const colonIdx = line.indexOf(":")
					if (colonIdx === -1) continue

					const field = line.slice(0, colonIdx)
					let value = line.slice(colonIdx + 1)
					if (value.startsWith(" ")) value = value.slice(1) // Remove leading space

					switch (field) {
						case "event":
							currentEvent.type = value
							break
						case "data":
							currentEvent.data = (currentEvent.data ?? "") + value
							break
						case "id":
							currentEvent.id = value
							break
					}
				}
			}
		} catch (err: any) {
			if (!aborted && err?.name !== "AbortError") {
				onError?.(err instanceof Error ? err : new Error(String(err)))
			}
		}
	})()

	return {
		disconnect: () => {
			aborted = true
			controller.abort()
		},
		jobId,
	}
}
