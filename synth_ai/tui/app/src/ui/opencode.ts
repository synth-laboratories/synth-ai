/**
 * OpenCode pane rendering and state management.
 */
import type { AppContext } from "../context"
import type { OpenCodeMessage } from "../types"

/**
 * Format messages for display in the OpenCode pane.
 */
function formatMessages(messages: OpenCodeMessage[], maxWidth: number): string[] {
  const lines: string[] = []

  if (messages.length === 0) {
    return [
      "No messages yet.",
      "",
      "Connect to an OpenCode session to start chatting.",
      "Press Ctrl+O to open the sessions modal.",
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

  // Calculate available dimensions
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const maxWidth = Math.max(20, cols - 50) // Account for jobs pane
  const maxLines = Math.max(5, rows - 15)

  // Update status
  if (appState.openCodeSessionId) {
    const session = snapshot.sessions.find((s) => s.session_id === appState.openCodeSessionId)
    if (session) {
      const health = snapshot.sessionHealthResults.get(session.session_id)
      let statusText = `Session: ${session.session_id}`
      if (session.is_local) {
        statusText += " [local]"
      }
      if (health?.healthy) {
        statusText += health.response_time_ms ? ` (${health.response_time_ms}ms)` : " (healthy)"
      }
      if (appState.openCodeIsProcessing) {
        statusText += " | Processing..."
      }
      ui.openCodeStatus.content = statusText
    } else {
      ui.openCodeStatus.content = `Session: ${appState.openCodeSessionId} | Not found`
    }
  } else {
    ui.openCodeStatus.content = "Not connected - Press Ctrl+O for sessions"
  }

  // Format and display messages
  const messageLines = formatMessages(appState.openCodeMessages, maxWidth)

  // Apply scroll offset
  const scrollOffset = Math.max(
    0,
    Math.min(appState.openCodeScrollOffset, messageLines.length - maxLines)
  )
  const visibleLines = messageLines.slice(scrollOffset, scrollOffset + maxLines)

  ui.openCodeMessagesText.content = visibleLines.join("\n")

  // Update input placeholder based on connection status
  if (appState.openCodeSessionId) {
    ui.openCodeInput.placeholder = "Type a message and press Enter..."
  } else {
    ui.openCodeInput.placeholder = "Connect to a session first (press 'o')"
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

  const messageLines = formatMessages(appState.openCodeMessages, maxWidth)
  const maxOffset = Math.max(0, messageLines.length - maxLines)

  appState.openCodeScrollOffset = Math.max(
    0,
    Math.min(appState.openCodeScrollOffset + delta, maxOffset)
  )

  renderOpenCodePane(ctx)
}

/**
 * Send a message to the OpenCode session.
 */
export async function sendOpenCodeMessage(ctx: AppContext): Promise<void> {
  const { appState, snapshot } = ctx.state
  const { ui } = ctx

  const content = ui.openCodeInput.value?.trim()
  if (!content) return

  if (!appState.openCodeSessionId) {
    // Show error in conversation area
    appState.openCodeMessages.push({
      id: `error-${Date.now()}`,
      role: "assistant",
      content: "Not connected to any session. Press Ctrl+O to open sessions and connect.",
      timestamp: new Date(),
    })
    snapshot.status = "Not connected - Press Ctrl+O for sessions"
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
  ui.openCodeInput.value = ""

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
  const { ui } = ctx
  if (!ui.openCodeInput.value) {
    ui.openCodeInput.value = ""
  }
  ui.openCodeInput.value += char
  ctx.render()
}

/**
 * Handle backspace for the OpenCode input field.
 */
export function handleOpenCodeBackspace(ctx: AppContext): void {
  const { ui } = ctx
  if (ui.openCodeInput.value && ui.openCodeInput.value.length > 0) {
    ui.openCodeInput.value = ui.openCodeInput.value.slice(0, -1)
    ctx.render()
  }
}
