/**
 * SSE client for real-time job updates from /api/jobs/stream
 */

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
	let aborted = false
	const controller = new AbortController()

	const url = `${process.env.SYNTH_BACKEND_URL}/api/jobs/stream?since_seq=${sinceSeq}`
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
				throw new Error(`SSE stream failed: HTTP ${res.status} ${res.statusText} - ${body.slice(0, 100)}`)
			}

			if (!res.body) {
				throw new Error("SSE stream: no response body")
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
								const data = JSON.parse(currentEvent.data) as JobStreamEvent
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
	}
}
