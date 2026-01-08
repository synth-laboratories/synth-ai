/**
 * SSE client for real-time job details updates from /api/prompt-learning/online/jobs/{job_id}/events/stream
 * Works for ALL job types: eval, learning, prompt-learning
 * 
 * Ported from feat/job-details branch.
 */

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
  let aborted = false
  const controller = new AbortController()

  // Use prompt-learning SSE endpoint (works for all jobs in learning_jobs table)
  const url = `${process.env.SYNTH_BACKEND_URL}/api/prompt-learning/online/jobs/${jobId}/events/stream?since_seq=${sinceSeq}`
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
        throw new Error(`Job details SSE stream failed: HTTP ${res.status} ${res.statusText} - ${body.slice(0, 100)}`)
      }

      if (!res.body) {
        throw new Error("Job details SSE stream: no response body")
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
                const data = JSON.parse(currentEvent.data) as JobDetailsStreamEvent
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
    } catch (err: unknown) {
      if (!aborted && (err as { name?: string })?.name !== "AbortError") {
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


