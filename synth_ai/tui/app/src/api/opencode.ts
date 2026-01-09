/**
 * OpenCode Event Streaming API.
 *
 * Provides functions for connecting to and subscribing to OpenCode
 * event streams via Server-Sent Events (SSE).
 */

import type { AppContext } from "../context"
import type { OpenCodeMessage } from "../types"

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

/** Message part from OpenCode */
export type OpenCodeMessagePart = {
  id: string
  messageID: string
  sessionID: string
  type: "tool" | "text" | "step-start" | "step-finish" | "subtask"
  state: {
    status: "pending" | "running" | "completed"
    title?: string
    input?: Record<string, any>
    output?: string
  }
  time?: {
    start?: number
    end?: number
  }
}

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
export async function sendPrompt(
  baseUrl: string,
  sessionId: string,
  prompt: string
): Promise<{ success: boolean; error?: string }> {
  try {
    const response = await fetch(`${baseUrl}/session/${sessionId}/message`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        parts: [
          { type: "text", text: prompt }
        ]
      }),
    })

    if (!response.ok) {
      const text = await response.text().catch(() => "")
      return { success: false, error: `HTTP ${response.status}: ${text}` }
    }

    return { success: true }
  } catch (err: any) {
    return { success: false, error: err?.message || "Unknown error" }
  }
}

/**
 * Process OpenCode event and update app state.
 */
export function processOpenCodeEvent(ctx: AppContext, event: OpenCodeEvent): void {
  const { appState } = ctx.state

  switch (event.type) {
    case "message.part.updated": {
      const part = event.properties.part as OpenCodeMessagePart
      if (!part) return

      // Find existing message or create new one
      const existingIdx = appState.openCodeMessages.findIndex(
        (m) => m.id === part.messageID
      )

      if (existingIdx >= 0) {
        // Update existing message
        const existing = appState.openCodeMessages[existingIdx]
        if (part.type === "text" && event.properties.delta) {
          existing.content += event.properties.delta
        } else if (part.type === "tool") {
          existing.toolStatus = part.state.status as any
          if (part.state.output) {
            existing.content = part.state.output
          }
        }
      } else {
        // Add new message
        const newMessage: OpenCodeMessage = {
          id: part.messageID,
          role: part.type === "tool" ? "tool" : "assistant",
          content: event.properties.delta || part.state.output || "",
          timestamp: new Date(),
          toolName: part.type === "tool" ? part.state.title : undefined,
          toolStatus: part.type === "tool" ? (part.state.status as any) : undefined,
        }
        appState.openCodeMessages.push(newMessage)
      }
      break
    }

    case "session.idle": {
      appState.openCodeIsProcessing = false
      break
    }

    case "session.error": {
      appState.openCodeIsProcessing = false
      const errorMsg = event.properties.error || "Unknown error"
      appState.openCodeMessages.push({
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `Error: ${errorMsg}`,
        timestamp: new Date(),
      })
      break
    }

    case "server.heartbeat": {
      // No action needed - just keep-alive
      break
    }
  }

  ctx.render()
}

/**
 * Add a user message to the conversation.
 */
export function addUserMessage(ctx: AppContext, content: string): void {
  const { appState } = ctx.state

  appState.openCodeMessages.push({
    id: `user-${Date.now()}`,
    role: "user",
    content,
    timestamp: new Date(),
  })

  ctx.render()
}

/**
 * Clear all OpenCode messages.
 */
export function clearOpenCodeMessages(ctx: AppContext): void {
  ctx.state.appState.openCodeMessages = []
  ctx.render()
}
