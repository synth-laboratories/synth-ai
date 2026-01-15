/**
 * OpenCode Event Streaming API.
 *
 * Provides functions for connecting to and subscribing to OpenCode
 * event streams via Server-Sent Events (SSE).
 */

import type { AppContext } from "../context"
import type { OpenCodeMessage, SessionHealthResult, SessionRecord } from "../types"
import { fetchWithTimeout, getRequestSignal, isAborted } from "../utils/request"
import { isAbortError } from "../utils/abort"
import { checkSessionHealth } from "./sessions"
import { log } from "../utils/log"
import { DEFAULT_OPENCODE_TIMEOUT_MS } from "../network"
import { connectJsonStream } from "./stream"

/** OpenCode event types */
export type OpenCodeEventType =
  | "message.part.updated"
  | "message.part.removed"
  | "message.updated"
  | "session.status"
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

export type LocalOpenCodeConnectResult =
  | { ok: true; session: SessionRecord; health: SessionHealthResult; aborted?: boolean }
  | { ok: false; error: string; health?: SessionHealthResult; aborted?: boolean }

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
  const connection = connectJsonStream<OpenCodeEvent>({
    url: url.toString(),
    includeScope: false,
    label: "opencode-events",
    onOpen: onConnect,
    onEvent: (event) => {
      if (!isActive) return
      onEvent(event)
    },
    onError: (error) => {
      if (!isActive) return
      onError?.(error)
    },
  })

  return {
    unsubscribe: () => {
      isActive = false
      connection.disconnect()
    },
    get isActive() {
      return isActive
    },
  }
}

function buildLocalSessionRecord(sessionId: string, opencodeUrl: string): SessionRecord {
  const now = new Date().toISOString()
  return {
    session_id: sessionId,
    container_id: "",
    state: "connected",
    mode: "interactive",
    model: "gpt-4o-mini",
    access_url: opencodeUrl,
    tunnel_url: null,
    opencode_url: opencodeUrl,
    health_url: `${opencodeUrl}/health`,
    created_at: now,
    connected_at: now,
    last_activity: now,
    error_message: null,
    metadata: {},
    is_local: true,
  }
}

function resolveOpenCodeWorkingDir(): string | null {
  const raw = (
    process.env.SYNTH_TUI_LAUNCH_CWD ||
    process.env.OPENCODE_WORKING_DIR ||
    process.env.INIT_CWD ||
    process.env.PWD ||
    process.cwd()
  ) as string
  const trimmed = raw.trim()
  return trimmed || null
}

export async function connectLocalOpenCodeSession(
  opencodeUrl: string,
  timeoutMs: number = 5000,
): Promise<LocalOpenCodeConnectResult> {
  const healthCheck = await checkSessionHealth({
    session_id: "local",
    container_id: "",
    state: "connecting",
    mode: "interactive",
    model: "gpt-4o-mini",
    access_url: opencodeUrl,
    tunnel_url: null,
    opencode_url: opencodeUrl,
    health_url: `${opencodeUrl}/health`,
    created_at: new Date().toISOString(),
    connected_at: null,
    last_activity: null,
    error_message: null,
    metadata: {},
    is_local: true,
  }, timeoutMs)

  if (isAborted()) {
    return { ok: false, error: "Cancelled", health: healthCheck, aborted: true }
  }

  if (!healthCheck.healthy) {
    return {
      ok: false,
      error: healthCheck.error || "OpenCode server not reachable",
      health: healthCheck,
    }
  }

  const managed = getRequestSignal()
  const workingDir = resolveOpenCodeWorkingDir()
  const url = workingDir
    ? `${opencodeUrl}/session?directory=${encodeURIComponent(workingDir)}`
    : `${opencodeUrl}/session`
  const start = Date.now()
  log("http", `→ POST ${url}`)
  try {
    const response = await fetchWithTimeout(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
      signal: managed.signal,
      timeoutMs: DEFAULT_OPENCODE_TIMEOUT_MS,
    })
    log("http", `← ${response.status} POST ${url} (${Date.now() - start}ms)`)

    if (!response.ok) {
      const errorText = await response.text().catch(() => "")
      return {
        ok: false,
        error: `Failed to create session: ${response.status} ${errorText}`.trim(),
        health: healthCheck,
      }
    }

    const sessionData = await response.json() as { id: string; title?: string }
    return {
      ok: true,
      session: buildLocalSessionRecord(sessionData.id, opencodeUrl),
      health: healthCheck,
    }
  } catch (err: any) {
    if (isAbortError(err)) {
      return { ok: false, error: "Cancelled", health: healthCheck, aborted: true }
    }
    log("http", `✗ POST ${url} - ${err?.message}`)
    return {
      ok: false,
      error: err?.message || "Failed to connect",
      health: healthCheck,
    }
  } finally {
    managed.dispose()
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
  const managed = getRequestSignal()
  const url = `${baseUrl}/session/${sessionId}/message`
  const start = Date.now()
  log("http", `→ POST ${url}`)
  try {
    const response = await fetchWithTimeout(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        parts: [
          { type: "text", text: prompt }
        ]
      }),
      signal: managed.signal,
      timeoutMs: DEFAULT_OPENCODE_TIMEOUT_MS,
    })
    log("http", `← ${response.status} POST ${url} (${Date.now() - start}ms)`)

    if (!response.ok) {
      const text = await response.text().catch(() => "")
      return { success: false, error: `HTTP ${response.status}: ${text}` }
    }

    return { success: true }
  } catch (err: any) {
    log("http", `✗ POST ${url} - ${err?.message}`)
    return { success: false, error: err?.message || "Unknown error" }
  } finally {
    managed.dispose()
  }
}

/**
 * Process OpenCode event and update app state.
 */
export function processOpenCodeEvent(ctx: AppContext, event: OpenCodeEvent): void {
  const { ui } = ctx.state
  const { setUi } = ctx

  switch (event.type) {
    case "message.part.updated": {
      const part = event.properties.part as OpenCodeMessagePart
      if (!part) return

      // Find existing message or create new one
      const existingIdx = ui.openCodeMessages.findIndex(
        (m) => m.id === part.messageID
      )

      if (existingIdx >= 0) {
        // Update existing message
        const existing = ui.openCodeMessages[existingIdx]
        if (part.type === "text" && event.properties.delta) {
          setUi("openCodeMessages", existingIdx, "content", `${existing.content}${event.properties.delta}`)
        } else if (part.type === "tool") {
          setUi("openCodeMessages", existingIdx, "toolStatus", part.state.status as any)
          if (part.state.output) {
            setUi("openCodeMessages", existingIdx, "content", part.state.output)
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
        setUi("openCodeMessages", (messages) => [...messages, newMessage])
      }
      break
    }

    case "session.idle": {
      setUi("openCodeIsProcessing", false)
      break
    }

    case "session.error": {
      setUi("openCodeIsProcessing", false)
      const errorMsg = event.properties.error || "Unknown error"
      const errorMessage: OpenCodeMessage = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `Error: ${errorMsg}`,
        timestamp: new Date(),
      }
      setUi("openCodeMessages", (messages) => [...messages, errorMessage])
      break
    }

    case "server.heartbeat": {
      // No action needed - just keep-alive
      break
    }
  }
}

/**
 * Add a user message to the conversation.
 */
export function addUserMessage(ctx: AppContext, content: string): void {
  const { setUi } = ctx
  const message: OpenCodeMessage = {
    id: `user-${Date.now()}`,
    role: "user",
    content,
    timestamp: new Date(),
  }
  setUi("openCodeMessages", (messages) => [...messages, message])
}

/**
 * Clear all OpenCode messages.
 */
export function clearOpenCodeMessages(ctx: AppContext): void {
  const { setUi } = ctx
  setUi("openCodeMessages", [])
}
