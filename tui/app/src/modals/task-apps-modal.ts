/**
 * Task Apps (Tunnels) detail modal controller.
 * Shows all tunnels with health status, hostname, port info.
 */
import type { AppContext } from "../context"
import type { TunnelRecord, TunnelHealthResult } from "../types"
import { copyToClipboard } from "../utils/clipboard"
import { createModalUI, clamp, wrapModalText, type ModalController } from "./base"
import { focusManager } from "../focus"

/**
 * Format detailed tunnel information for the modal.
 */
function formatTunnelDetails(
  tunnels: TunnelRecord[],
  healthResults: Map<string, TunnelHealthResult>,
  selectedIndex: number
): string {
  // Filter out deleted tunnels - only show active ones
  const activeTunnels = tunnels.filter((t) => t.status === "active" && !t.deleted_at)
  
  if (activeTunnels.length === 0) {
    return "No active task apps (tunnels).\n\nTask apps are Cloudflare managed tunnels that expose\nlocal APIs to the internet for remote execution.\n\nPress 'q' to close."
  }

  const lines: string[] = []

  activeTunnels.forEach((tunnel, idx) => {
    const health = healthResults.get(tunnel.id)
    const isSelected = idx === selectedIndex

    // Health indicator
    let healthIcon = "?"
    let healthText = "checking..."
    if (health) {
      if (health.healthy) {
        healthIcon = "\u2713"  // checkmark
        healthText = health.response_time_ms != null 
          ? `Healthy (${health.response_time_ms}ms)` 
          : "Healthy"
      } else {
        healthIcon = "\u2717"  // X
        healthText = health.error?.slice(0, 40) || "Unhealthy"
      }
    }

    // Extract port from hostname (task-PORT-PID format) or use local_port
    const portMatch = tunnel.hostname.match(/task-(\d+)-\d+/)
    const displayPort = portMatch ? portMatch[1] : tunnel.local_port?.toString() || "?"

    // Selection indicator
    const prefix = isSelected ? "> " : "  "
    const hostname = tunnel.hostname.replace(/^https?:\/\//, "")

    // Main line - truncate long hostnames
    const shortHost = hostname.length > 50 ? hostname.slice(0, 47) + "..." : hostname
    lines.push(`${prefix}[${healthIcon}] ${shortHost}`)
    lines.push(`    Port: ${displayPort} | Status: ${healthText}`)
    lines.push(`    Local: ${tunnel.local_host}:${tunnel.local_port}`)
    if (tunnel.created_at) {
      const created = new Date(tunnel.created_at)
      const createdStr = created.toLocaleString()
      lines.push(`    Created: ${createdStr}`)
    }
    if (tunnel.org_name) {
      lines.push(`    Org: ${tunnel.org_name}`)
    }
    lines.push("")
  })

  return lines.join("\n")
}

export function createTaskAppsModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  updateContent: () => void
  copyHostname: () => Promise<void>
} {
  const { renderer } = ctx
  const { appState, snapshot } = ctx.state

  const modal = createModalUI(renderer, {
    id: "task-apps-modal",
    width: 90,
    height: 20,
    borderColor: "#06b6d4",
    titleColor: "#06b6d4",
    zIndex: 8,
  })

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "task-apps-modal",
        handleKey,
      })
      modal.center()
    } else {
      focusManager.pop("task-apps-modal")
      modal.setContent("")
    }
    modal.setVisible(visible)
  }

  function updateContent(): void {
    if (!modal.visible) return

    const raw = formatTunnelDetails(
      snapshot.tunnels,
      snapshot.tunnelHealthResults,
      appState.taskAppsModalSelectedIndex || 0
    )
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, (typeof process.stdout?.rows === "number" ? process.stdout.rows : 40) - 12)

    appState.taskAppsModalOffset = clamp(appState.taskAppsModalOffset || 0, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(appState.taskAppsModalOffset, appState.taskAppsModalOffset + maxLines)

    const tunnelCount = snapshot.tunnels.length
    modal.setTitle(`Task Apps (${tunnelCount} tunnel${tunnelCount !== 1 ? "s" : ""})`)
    modal.setContent(visible.join("\n"))
    modal.setHint(
      wrapped.length > maxLines
        ? `[${appState.taskAppsModalOffset + 1}-${appState.taskAppsModalOffset + visible.length}/${wrapped.length}] j/k scroll | y copy hostname | q close`
        : "j/k select | y copy hostname | q close"
    )
  }

  function move(delta: number): void {
    // Only count active tunnels for selection
    const activeTunnels = snapshot.tunnels.filter((t) => t.status === "active" && !t.deleted_at)
    const maxIndex = Math.max(0, activeTunnels.length - 1)
    appState.taskAppsModalSelectedIndex = clamp(
      (appState.taskAppsModalSelectedIndex || 0) + delta,
      0,
      maxIndex
    )
    updateContent()
  }

  function open(): void {
    appState.taskAppsModalOffset = 0
    appState.taskAppsModalSelectedIndex = 0
    toggle(true)
    updateContent()
  }

  async function copyHostname(): Promise<void> {
    const activeTunnels = snapshot.tunnels.filter((t) => t.status === "active" && !t.deleted_at)
    const selectedIndex = appState.taskAppsModalSelectedIndex || 0
    const tunnel = activeTunnels[selectedIndex]
    if (tunnel) {
      const hostname = tunnel.hostname.replace(/^https?:\/\//, "")
      const url = `https://${hostname}`
      await copyToClipboard(url)
      snapshot.status = `Copied: ${url}`
      ctx.render()
    }
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false

    if (key.name === "up" || key.name === "k") {
      move(-1)
      return true
    }
    if (key.name === "down" || key.name === "j") {
      move(1)
      return true
    }
    if (key.name === "y") {
      void copyHostname()
      return true
    }
    if (key.name === "return" || key.name === "enter" || key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true // consume all keys when modal is open
  }

  const controller = {
    get isVisible() {
      return modal.visible
    },
    toggle,
    open,
    move,
    updateContent,
    copyHostname,
    handleKey,
  }

  return controller
}
