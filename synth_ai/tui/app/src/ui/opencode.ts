/**
 * OpenCode pane rendering and state management.
 *
 * NOTE: This file contains stub implementations until the OpenCode pane
 * UI components are added to layout.ts (Phase 6 of the integration plan).
 */
import type { AppContext } from "../context"
import type { OpenCodeMessage } from "../types"

/**
 * Format messages for display in the OpenCode pane.
 */
export function formatMessages(messages: OpenCodeMessage[], maxWidth: number, isConnected: boolean = false): string[] {
  const lines: string[] = []

  if (messages.length === 0) {
    if (isConnected) {
      return [
        "Connected! Ready to chat.",
        "",
        "Type your message below and press Enter to send.",
      ]
    }
    return [
      "No messages yet.",
      "",
      "Connect to an OpenCode session to start chatting.",
      "Press Shift+O to open the sessions modal.",
    ]
  }

  for (const msg of messages) {
    // Format timestamp
    const time = msg.timestamp.toLocaleTimeString()

    if (msg.role === "user") {
      lines.push(`[${time}] You:`)
      // Wrap content
      const contentLines = wrapText(msg.content, maxWidth - 2)
      for (const line of contentLines) {
        lines.push(`  ${line}`)
      }
      lines.push("")
    } else if (msg.role === "tool") {
      const status = msg.toolStatus || "running"
      const statusIcon = status === "completed" ? "\u2713" : status === "failed" ? "\u2717" : "\u21BB"
      lines.push(`[${time}] [${statusIcon}] ${msg.toolName || "Tool"}:`)
      if (msg.content) {
        const contentLines = wrapText(msg.content, maxWidth - 2)
        for (const line of contentLines) {
          lines.push(`  ${line}`)
        }
      }
      lines.push("")
    } else {
      // assistant
      lines.push(`[${time}] Agent:`)
      const contentLines = wrapText(msg.content, maxWidth - 2)
      for (const line of contentLines) {
        lines.push(`  ${line}`)
      }
      lines.push("")
    }
  }

  return lines
}

/**
 * Wrap text to fit within max width.
 */
function wrapText(text: string, maxWidth: number): string[] {
  const lines: string[] = []
  const paragraphs = text.split("\n")

  for (const para of paragraphs) {
    if (para.length <= maxWidth) {
      lines.push(para)
      continue
    }

    // Word wrap
    const words = para.split(" ")
    let currentLine = ""

    for (const word of words) {
      if (currentLine.length === 0) {
        currentLine = word
      } else if (currentLine.length + 1 + word.length <= maxWidth) {
        currentLine += " " + word
      } else {
        lines.push(currentLine)
        currentLine = word
      }
    }

    if (currentLine.length > 0) {
      lines.push(currentLine)
    }
  }

  return lines
}

/**
 * Render the OpenCode pane content.
 */
export function renderOpenCodePane(ctx: AppContext): void {
  const { ui } = ctx
  const { appState, snapshot } = ctx.state

  // Calculate dimensions
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const maxWidth = Math.max(20, cols - 50)
  const maxLines = Math.max(5, rows - 15)

  // Format and display messages
  const isConnected = !!appState.openCodeSessionId
  const messageLines = formatMessages(appState.openCodeMessages, maxWidth, isConnected)
  const visibleLines = messageLines.slice(
    appState.openCodeScrollOffset,
    appState.openCodeScrollOffset + maxLines
  )
  ui.openCodeMessagesText.content = visibleLines.join("\n")

  // Update input field
  ui.openCodeInputText.content = appState.openCodeInputValue || ""

  // Update status text based on session state
  if (appState.openCodeSessionId) {
    const session = snapshot.sessions.find((s) => s.session_id === appState.openCodeSessionId)
    if (session) {
      const url = session.opencode_url || session.access_url || "unknown"
      ui.openCodeStatusText.content = appState.openCodeIsProcessing
        ? `Processing... | Session: ${session.session_id}`
        : `Connected to ${url} | Session: ${session.session_id.slice(0, 20)}...`
      ui.openCodeStatusText.fg = "#22c55e"  // green for connected
    } else {
      ui.openCodeStatusText.content = `Session ${appState.openCodeSessionId} not found`
      ui.openCodeStatusText.fg = "#f97316"  // orange for warning
    }
  } else {
    ui.openCodeStatusText.content = "Not connected - Press Shift+O for sessions"
    ui.openCodeStatusText.fg = "#94a3b8"  // gray for disconnected
  }
}

/**
 * Scroll the OpenCode messages pane.
 */
export function scrollOpenCode(ctx: AppContext, delta: number): void {
  const { appState } = ctx.state

  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const maxWidth = Math.max(20, cols - 50)
  const maxLines = Math.max(5, rows - 15)

  const isConnected = !!appState.openCodeSessionId
  const messageLines = formatMessages(appState.openCodeMessages, maxWidth, isConnected)
  const maxOffset = Math.max(0, messageLines.length - maxLines)

  appState.openCodeScrollOffset = Math.max(
    0,
    Math.min(appState.openCodeScrollOffset + delta, maxOffset)
  )

  renderOpenCodePane(ctx)
}

/**
 * Send a message to the OpenCode session.
 *
 * STUB: Input handling not yet implemented
 */
export async function sendOpenCodeMessage(ctx: AppContext): Promise<void> {
  const { appState, snapshot } = ctx.state

  // TODO: Get content from input field when UI is added
  const content = appState.openCodeInputValue?.trim()
  if (!content) return

  if (!appState.openCodeSessionId) {
    appState.openCodeMessages.push({
      id: `error-${Date.now()}`,
      role: "assistant",
      content: "Not connected to any session. Press Shift+O to open sessions and connect.",
      timestamp: new Date(),
    })
    snapshot.status = "Not connected - Press Shift+O for sessions"
    renderOpenCodePane(ctx)
    ctx.render()
    return
  }

  // Find session
  const session = snapshot.sessions.find((s) => s.session_id === appState.openCodeSessionId)
  if (!session) {
    snapshot.status = "Session not found"
    ctx.render()
    return
  }

  // Add user message to conversation
  appState.openCodeMessages.push({
    id: `user-${Date.now()}`,
    role: "user",
    content,
    timestamp: new Date(),
  })

  // Clear input
  appState.openCodeInputValue = ""

  // Set processing state
  appState.openCodeIsProcessing = true
  renderOpenCodePane(ctx)

  // Send to OpenCode server
  try {
    const { sendPrompt } = await import("../api/opencode")
    const baseUrl = session.opencode_url || session.access_url
    if (!baseUrl) {
      throw new Error("No URL for session")
    }

    const result = await sendPrompt(baseUrl, appState.openCodeSessionId, content)
    if (!result.success) {
      appState.openCodeMessages.push({
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `Error: ${result.error || "Failed to send message"}`,
        timestamp: new Date(),
      })
      appState.openCodeIsProcessing = false
    }
    // Response will come via SSE events
  } catch (err: any) {
    appState.openCodeMessages.push({
      id: `error-${Date.now()}`,
      role: "assistant",
      content: `Error: ${err?.message || "Failed to send message"}`,
      timestamp: new Date(),
    })
    appState.openCodeIsProcessing = false
  }

  renderOpenCodePane(ctx)
}

/**
 * Handle character input for the OpenCode input field.
 */
export function handleOpenCodeInput(ctx: AppContext, char: string): void {
  const { appState } = ctx.state
  if (!appState.openCodeInputValue) {
    appState.openCodeInputValue = ""
  }
  appState.openCodeInputValue += char
  renderOpenCodePane(ctx)
  ctx.render()
}

/**
 * Handle backspace for the OpenCode input field.
 */
export function handleOpenCodeBackspace(ctx: AppContext): void {
  const { appState } = ctx.state
  if (appState.openCodeInputValue && appState.openCodeInputValue.length > 0) {
    appState.openCodeInputValue = appState.openCodeInputValue.slice(0, -1)
    renderOpenCodePane(ctx)
    ctx.render()
  }
}
