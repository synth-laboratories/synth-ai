/**
 * Interactive Sessions modal controller.
 * Shows all OpenCode sessions with connection status, URLs, and actions.
 */
import type { AppContext } from "../context"
import type { SessionRecord, SessionHealthResult } from "../types"
import { blurForModal, restoreFocusFromModal } from "../ui/panes"
import { copyToClipboard } from "../utils/clipboard"
import { clamp, wrapModalText, type ModalController } from "./base"
import { fetchSessions, disconnectSession, checkSessionHealth } from "../api/sessions"

/**
 * Format session details for the modal.
 */
function formatSessionDetails(
  sessions: SessionRecord[],
  healthResults: Map<string, SessionHealthResult>,
  selectedIndex: number,
  openCodeUrl: string | null
): string {
  // Filter to connected/connecting sessions
  const activeSessions = sessions.filter((s) =>
    s.state === "connected" || s.state === "connecting" || s.state === "reconnecting"
  )

  const serverUrl = openCodeUrl || "(not started)"

  if (activeSessions.length === 0) {
    return `No active OpenCode sessions.

Interactive sessions connect to local or remote OpenCode servers
for real-time agent interaction.

OpenCode server: ${serverUrl}

Quick connect:
  Press 'c' to connect to the local OpenCode server
  Press 'C' to connect with custom URL

Press 'q' to close.`
  }

  const lines: string[] = []

  activeSessions.forEach((session, idx) => {
    const health = healthResults.get(session.session_id)
    const isSelected = idx === selectedIndex

    // Health/state indicator
    let stateIcon = "?"
    let stateText: string = session.state
    if (session.state === "connected") {
      if (health) {
        if (health.healthy) {
          stateIcon = "\u2713"  // checkmark
          stateText = health.response_time_ms != null
            ? `Connected (${health.response_time_ms}ms)`
            : "Connected"
        } else {
          stateIcon = "\u2717"  // X
          stateText = health.error?.slice(0, 30) || "Unhealthy"
        }
      } else {
        stateIcon = "\u2713"
        stateText = "Connected"
      }
    } else if (session.state === "connecting" || session.state === "reconnecting") {
      stateIcon = "\u21BB"  // refresh
      stateText = session.state
    } else if (session.state === "error") {
      stateIcon = "\u2717"
      stateText = session.error_message?.slice(0, 30) || "Error"
    }

    // Selection indicator
    const prefix = isSelected ? "> " : "  "

    // Local indicator
    const localTag = session.is_local ? " [local]" : ""

    // Main line
    lines.push(`${prefix}[${stateIcon}] ${session.session_id}${localTag}`)
    lines.push(`    State: ${stateText}`)
    lines.push(`    Mode: ${session.mode} | Model: ${session.model || "default"}`)

    // URLs
    if (session.opencode_url) {
      const shortUrl = session.opencode_url.length > 50
        ? session.opencode_url.slice(0, 47) + "..."
        : session.opencode_url
      lines.push(`    URL: ${shortUrl}`)
    }
    if (session.tunnel_url && session.tunnel_url !== session.opencode_url) {
      lines.push(`    Tunnel: ${session.tunnel_url}`)
    }

    // Timestamps
    if (session.connected_at) {
      const connectedAt = new Date(session.connected_at)
      lines.push(`    Connected: ${connectedAt.toLocaleString()}`)
    }
    if (session.last_activity) {
      const lastActivity = new Date(session.last_activity)
      lines.push(`    Last activity: ${lastActivity.toLocaleString()}`)
    }

    lines.push("")
  })

  return lines.join("\n")
}

export function createSessionsModal(ctx: AppContext): ModalController & {
  open: () => Promise<void>
  move: (delta: number) => void
  updateContent: () => void
  copyUrl: () => Promise<void>
  connectLocalSession: () => Promise<void>
  disconnectSelected: () => Promise<void>
  refreshHealth: () => Promise<void>
  selectSession: () => void
} {
  const { ui, renderer } = ctx
  const { appState, snapshot } = ctx.state

  // Local state for sessions
  let sessions: SessionRecord[] = []
  let healthResults: Map<string, SessionHealthResult> = new Map()
  let selectedIndex = 0
  let scrollOffset = 0

  function toggle(visible: boolean): void {
    ui.sessionsModalVisible = visible
    ui.sessionsModalBox.visible = visible
    ui.sessionsModalTitle.visible = visible
    ui.sessionsModalText.visible = visible
    ui.sessionsModalHint.visible = visible
    if (visible) {
      blurForModal(ctx)
    } else {
      ui.sessionsModalText.content = ""
      restoreFocusFromModal(ctx)
    }
    renderer.requestRender()
  }

  function updateContent(): void {
    if (!ui.sessionsModalVisible) return

    const activeSessions = sessions.filter((s) =>
      s.state === "connected" || s.state === "connecting" || s.state === "reconnecting"
    )

    const raw = formatSessionDetails(sessions, healthResults, selectedIndex, appState.openCodeUrl)
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, (typeof process.stdout?.rows === "number" ? process.stdout.rows : 40) - 12)

    scrollOffset = clamp(scrollOffset, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(scrollOffset, scrollOffset + maxLines)

    const sessionCount = activeSessions.length
    ui.sessionsModalTitle.content = `OpenCode Sessions (${sessionCount} active)`
    ui.sessionsModalText.content = visible.join("\n")
    ui.sessionsModalHint.content =
      wrapped.length > maxLines
        ? `[${scrollOffset + 1}-${scrollOffset + visible.length}/${wrapped.length}] j/k select | c connect local | d disconnect | y copy URL | enter select | q close`
        : "j/k select | c connect local | d disconnect | y copy URL | enter select | q close"

    renderer.requestRender()
  }

  function move(delta: number): void {
    const activeSessions = sessions.filter((s) =>
      s.state === "connected" || s.state === "connecting" || s.state === "reconnecting"
    )
    const maxIndex = Math.max(0, activeSessions.length - 1)
    selectedIndex = clamp(selectedIndex + delta, 0, maxIndex)
    updateContent()
  }

  async function open(): Promise<void> {
    scrollOffset = 0
    selectedIndex = 0
    toggle(true)

    // Fetch sessions
    snapshot.status = "Loading sessions..."
    ctx.render()

    try {
      sessions = await fetchSessions()
      snapshot.sessions = sessions
      updateContent()

      // Check health in background
      void refreshHealth()
    } catch (err: any) {
      snapshot.lastError = err?.message || "Failed to load sessions"
      updateContent()
    }
  }

  async function refreshHealth(): Promise<void> {
    const activeSessions = sessions.filter((s) =>
      s.state === "connected" || s.state === "connecting"
    )

    for (const session of activeSessions) {
      const result = await checkSessionHealth(session)
      healthResults.set(session.session_id, result)
      snapshot.sessionHealthResults.set(session.session_id, result)
      updateContent()
    }
  }

  async function copyUrl(): Promise<void> {
    const activeSessions = sessions.filter((s) =>
      s.state === "connected" || s.state === "connecting" || s.state === "reconnecting"
    )
    const session = activeSessions[selectedIndex]
    if (session) {
      const url = session.opencode_url || session.access_url || ""
      if (url) {
        await copyToClipboard(url)
        snapshot.status = `Copied: ${url}`
        ctx.render()
      }
    }
  }

  async function connectLocalSession(): Promise<void> {
    const opencode_url = appState.openCodeUrl

    if (!opencode_url) {
      snapshot.lastError = "OpenCode server not started"
      snapshot.status = "No OpenCode server URL available - server may not be running"
      ctx.render()
      return
    }

    snapshot.status = `Connecting to OpenCode at ${opencode_url}...`
    ctx.render()

    try {
      // First check if OpenCode server is reachable
      const healthCheck = await checkSessionHealth({
        session_id: "local",
        container_id: "",
        state: "connecting",
        mode: "interactive",
        model: "gpt-4o-mini",
        access_url: opencode_url,
        tunnel_url: null,
        opencode_url: opencode_url,
        health_url: `${opencode_url}/health`,
        created_at: new Date().toISOString(),
        connected_at: null,
        last_activity: null,
        error_message: null,
        metadata: {},
        is_local: true,
      })

      if (!healthCheck.healthy) {
        snapshot.lastError = healthCheck.error || "OpenCode server not reachable"
        snapshot.status = `Connection failed - is OpenCode running at ${opencode_url}?`
        ctx.render()
        return
      }

      // Create a REAL session on the OpenCode server
      snapshot.status = `Creating session on OpenCode...`
      ctx.render()

      const dir = ctx.state.appState.opencodeWorkingDir
      const sessionCreateUrl = dir
        ? `${opencode_url}/session?directory=${encodeURIComponent(dir)}`
        : `${opencode_url}/session`
      const createResponse = await fetch(sessionCreateUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      })

      if (!createResponse.ok) {
        const errorText = await createResponse.text().catch(() => "")
        snapshot.lastError = `Failed to create session: ${createResponse.status} ${errorText}`
        snapshot.status = "Session creation failed"
        ctx.render()
        return
      }

      const sessionData = await createResponse.json() as { id: string; title?: string }
      const sessionId = sessionData.id

      const localSession: SessionRecord = {
        session_id: sessionId,
        container_id: "",
        state: "connected",
        mode: "interactive",
        model: "gpt-4o-mini",
        access_url: opencode_url,
        tunnel_url: null,
        opencode_url: opencode_url,
        health_url: `${opencode_url}/health`,
        created_at: new Date().toISOString(),
        connected_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
        error_message: null,
        metadata: {},
        is_local: true,
      }

      // Add to sessions list
      sessions = [localSession, ...sessions.filter(s => s.session_id !== sessionId)]
      snapshot.sessions = sessions
      healthResults.set(sessionId, healthCheck)
      snapshot.sessionHealthResults.set(sessionId, healthCheck)

      // Set as active session
      appState.openCodeSessionId = sessionId
      snapshot.status = `Connected to OpenCode at ${opencode_url} | Session: ${sessionId}`
      updateContent()
      ctx.render()
    } catch (err: any) {
      snapshot.lastError = err?.message || "Failed to connect"
      snapshot.status = "Connection failed"
      ctx.render()
    }
  }

  async function disconnectSelected(): Promise<void> {
    const activeSessions = sessions.filter((s) =>
      s.state === "connected" || s.state === "connecting" || s.state === "reconnecting"
    )
    const session = activeSessions[selectedIndex]
    if (!session) return

    snapshot.status = `Disconnecting ${session.session_id}...`
    ctx.render()

    try {
      const result = await disconnectSession(session.session_id)
      if (result.disconnected) {
        snapshot.status = `Disconnected from ${session.session_id}`
        // Clear active session if it was the disconnected one
        if (appState.openCodeSessionId === session.session_id) {
          appState.openCodeSessionId = null
        }
        // Refresh sessions list
        sessions = await fetchSessions()
        snapshot.sessions = sessions
        selectedIndex = Math.max(0, selectedIndex - 1)
      } else {
        snapshot.status = "Disconnect failed"
      }
      updateContent()
      ctx.render()
    } catch (err: any) {
      snapshot.lastError = err?.message || "Failed to disconnect"
      snapshot.status = "Disconnect failed"
      ctx.render()
    }
  }

  function selectSession(): void {
    const activeSessions = sessions.filter((s) =>
      s.state === "connected" || s.state === "connecting" || s.state === "reconnecting"
    )
    const session = activeSessions[selectedIndex]
    if (session) {
      appState.openCodeSessionId = session.session_id
      // Ensure the session is in snapshot.sessions for downstream consumers.
      if (!snapshot.sessions.find((s) => s.session_id === session.session_id)) {
        snapshot.sessions.push(session)
      }
      snapshot.status = `Selected session: ${session.session_id}`
      toggle(false)
      ctx.render()
    }
  }

  function handleKey(key: any): boolean {
    if (!ui.sessionsModalVisible) return false

    if (key.name === "up" || key.name === "k") {
      move(-1)
      return true
    }
    if (key.name === "down" || key.name === "j") {
      move(1)
      return true
    }
    if (key.name === "y") {
      void copyUrl()
      return true
    }
    if (key.name === "c" && !key.shift) {
      void connectLocalSession()
      return true
    }
    if (key.name === "d") {
      void disconnectSelected()
      return true
    }
    if (key.name === "r") {
      void open()
      return true
    }
    if (key.name === "return" || key.name === "enter") {
      selectSession()
      return true
    }
    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true
  }

  return {
    get isVisible() {
      return ui.sessionsModalVisible
    },
    toggle,
    open,
    move,
    updateContent,
    copyUrl,
    connectLocalSession,
    disconnectSelected,
    refreshHealth,
    selectSession,
    handleKey,
  }
}
