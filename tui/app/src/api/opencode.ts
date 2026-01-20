/**
 * OpenCode Event Streaming API.
 *
 * Provides functions for connecting to and subscribing to OpenCode
 * event streams via Server-Sent Events (SSE).
 */

/** OpenCode event types */
export type OpenCodeEventType =
  | "message.part.updated"
  | "message.part.removed"
  | "message.updated"
  | "session.idle"
  | "session.error"
  | "session.created"
  | "session.deleted"
  | "session.updated"
  | "server.connected"
  | "server.heartbeat"
  | "permission.asked"
  | "file.edited"
  | "pty.created"
  | "pty.updated"
  | "pty.exited"
  | "pty.deleted"

/** OpenCode SSE event */
export type OpenCodeEvent = {
  type: OpenCodeEventType
  properties: Record<string, any>
}

/** Event subscription handle */
export type EventSubscription = {
  /** Unsubscribe from events */
  unsubscribe: () => void
  /** Whether subscription is active */
  isActive: boolean
}

/**
 * Create an SSE connection to OpenCode event stream.
 * Uses fetch() streaming since Bun doesn't have native EventSource.
 *
 * @param baseUrl - OpenCode server URL (e.g., http://localhost:3000)
 * @param directory - Optional directory to scope events to
 * @param onEvent - Callback for received events
 * @param onError - Callback for errors
 * @param onConnect - Callback when connection established
 */
export function subscribeToOpenCodeEvents(
  baseUrl: string,
  options: {
    directory?: string
    onEvent: (event: OpenCodeEvent) => void
    onError?: (error: Error) => void
    onConnect?: () => void
  }
): EventSubscription {
  const { directory, onEvent, onError, onConnect } = options

  // Build URL with optional directory query param
  const url = new URL("/event", baseUrl)
  if (directory) {
    url.searchParams.set("directory", directory)
  }

  let isActive = true
  let abortController: AbortController | null = new AbortController()

  // Start the SSE connection using fetch streaming
  ;(async () => {
    try {
      const response = await fetch(url.toString(), {
        headers: {
          Accept: "text/event-stream",
        },
        signal: abortController?.signal,
      })

      if (!response.ok) {
        throw new Error(`SSE connection failed: ${response.status}`)
      }

      if (!response.body) {
        throw new Error("No response body for SSE stream")
      }

      if (onConnect) onConnect()

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (isActive) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Process complete SSE messages (each ends with \n\n)
        const lines = buffer.split("\n")
        buffer = lines.pop() || "" // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6)
            try {
              const event = JSON.parse(data) as OpenCodeEvent
              onEvent(event)
            } catch {
              // Ignore parse errors for heartbeats etc
            }
          }
        }
      }
    } catch (err: any) {
      if (!isActive) return // Ignore errors after unsubscribe
      if (err.name === "AbortError") return // Expected on unsubscribe
      if (onError) {
        onError(err)
      }
    }
  })()

  return {
    unsubscribe: () => {
      isActive = false
      if (abortController) {
        abortController.abort()
        abortController = null
      }
    },
    get isActive() {
      return isActive
    },
  }
}

/**
 * Send a prompt to OpenCode session.
 *
 * This sends a chat message to the OpenCode server for processing.
 *
 * @param baseUrl - OpenCode server URL
 * @param sessionId - Session ID to send to
 * @param prompt - The prompt text
 */
