/**
 * Log rendering + navigation helpers for deployment logs pane.
 */
import { TextRenderable } from "@opentui/core"
import type { AppContext } from "../context"
import type { DeploymentLog, LogSource } from "../types"

/** Color scheme for log sources */
const SOURCE_COLORS: Record<LogSource, string> = {
  uvicorn: "#22c55e",    // Green
  cloudflare: "#3b82f6", // Blue
  app: "#f59e0b",        // Amber
}

/** Color scheme for log levels */
const LEVEL_COLORS: Record<string, string> = {
  ERROR: "#ef4444",
  WARNING: "#f59e0b",
  INFO: "#e2e8f0",
  DEBUG: "#94a3b8",
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/** Get layout metrics for logs pane */
export function getLogsLayoutMetrics(_ctx: AppContext): {
  visibleCount: number
} {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  // Reserve space for header, tabs, status, footer, and box borders
  const available = Math.max(1, rows - 16)
  return { visibleCount: available }
}

/** Get filtered logs from the active deployment */
function getFilteredLogs(ctx: AppContext): DeploymentLog[] {
  const { snapshot, appState } = ctx.state
  const deploymentId = appState.logsActiveDeploymentId
  if (!deploymentId) return []

  const deployment = snapshot.deployments.get(deploymentId)
  if (!deployment) return []

  return deployment.logs.filter(
    (entry): entry is DeploymentLog =>
      entry.type === "log" && appState.logsSourceFilter.has(entry.source)
  )
}

/** Format a timestamp for display */
function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toISOString().slice(11, 23)
}

/** Convert hex color to RGB for ANSI escape codes */
function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `${r};${g};${b}`
}

/** Format a log entry for display */
function formatLogEntry(log: DeploymentLog, maxWidth: number): string {
  const timestamp = formatTimestamp(log.timestamp)
  const sourceColor = SOURCE_COLORS[log.source] || "#94a3b8"

  // Format: [HH:MM:SS.mmm] [SOURCE    ] message
  const prefix = `\x1b[90m${timestamp}\x1b[0m \x1b[38;2;${hexToRgb(sourceColor)}m[${log.source.padEnd(10)}]\x1b[0m `

  // Calculate available width for message (accounting for ANSI codes in prefix)
  const prefixVisibleLength = timestamp.length + 1 + 12 + 1 // timestamp + space + [source] + space
  const messageMaxLen = maxWidth - prefixVisibleLength - 2

  let message = log.message
  if (message.length > messageMaxLen && messageMaxLen > 3) {
    message = message.slice(0, messageMaxLen - 3) + "..."
  }

  return prefix + message
}

/** Render the logs pane */
export function renderLogs(ctx: AppContext): void {
  const { ui, renderer } = ctx
  const { snapshot, appState } = ctx.state

  // Get active deployment
  const deploymentId = appState.logsActiveDeploymentId
  const deployment = deploymentId ? snapshot.deployments.get(deploymentId) : null

  // Get filtered logs
  const filteredLogs = getFilteredLogs(ctx)

  if (!deployment || filteredLogs.length === 0) {
    ui.logsContent.visible = false
    ui.logsEmptyText.visible = true

    if (!deployment) {
      // Check if there are any deployments at all
      if (snapshot.deployments.size === 0) {
        ui.logsEmptyText.content = "No active deployments.\n\nPress 'n' to deploy a LocalAPI."
      } else {
        ui.logsEmptyText.content = "Select a deployment to view logs."
      }
    } else {
      ui.logsEmptyText.content = "Waiting for logs..."
    }
    return
  }

  ui.logsEmptyText.visible = false
  ui.logsContent.visible = true

  const { visibleCount } = getLogsLayoutMetrics(ctx)
  const total = filteredLogs.length

  // Handle tail mode (auto-scroll to latest)
  if (appState.logsTailMode && total > 0) {
    appState.logsWindowStart = Math.max(0, total - visibleCount)
    appState.logsSelectedIndex = total - 1
  }

  // Clamp selection and window
  appState.logsSelectedIndex = clamp(appState.logsSelectedIndex, 0, Math.max(0, total - 1))
  appState.logsWindowStart = clamp(
    appState.logsWindowStart,
    0,
    Math.max(0, total - visibleCount)
  )

  // Adjust window to keep selection visible
  if (appState.logsSelectedIndex < appState.logsWindowStart) {
    appState.logsWindowStart = appState.logsSelectedIndex
  } else if (appState.logsSelectedIndex >= appState.logsWindowStart + visibleCount) {
    appState.logsWindowStart = appState.logsSelectedIndex - visibleCount + 1
  }

  // Get visible logs
  const visibleLogs = filteredLogs.slice(
    appState.logsWindowStart,
    appState.logsWindowStart + visibleCount
  )

  // Clear existing entries
  for (const entry of ui.logEntries) {
    ui.logsContent.remove(entry.text.id)
  }
  ui.logEntries = []

  // Get terminal width for formatting
  const termWidth = typeof process.stdout?.columns === "number" ? process.stdout.columns : 80
  const maxWidth = termWidth - 4 // Account for box borders

  // Render visible log lines
  visibleLogs.forEach((log, index) => {
    const globalIndex = appState.logsWindowStart + index
    const isSelected = globalIndex === appState.logsSelectedIndex && !appState.logsTailMode

    const content = formatLogEntry(log, maxWidth)
    const levelColor = LEVEL_COLORS[log.level || "INFO"] || "#e2e8f0"

    const text = new TextRenderable(renderer, {
      id: `log-entry-${index}`,
      content: content,
      fg: isSelected ? "#ffffff" : levelColor,
      bg: isSelected ? "#1e293b" : undefined,
    })

    ui.logsContent.add(text)
    ui.logEntries.push({ text })
  })

  // Update logs box title to show filter status and position
  const activeFilters = Array.from(appState.logsSourceFilter).join(", ")
  const position = total > visibleCount ? ` [${appState.logsWindowStart + 1}-${Math.min(appState.logsWindowStart + visibleCount, total)}/${total}]` : ""
  const tailIndicator = appState.logsTailMode ? " [TAIL]" : ""
  ui.logsBox.title = `Logs (${activeFilters})${position}${tailIndicator}`
}

/** Move log selection up or down */
export function moveLogSelection(ctx: AppContext, delta: number): void {
  const filteredLogs = getFilteredLogs(ctx)
  if (!filteredLogs.length) return

  const { appState } = ctx.state

  // Disable tail mode when manually navigating
  appState.logsTailMode = false

  appState.logsSelectedIndex = clamp(
    appState.logsSelectedIndex + delta,
    0,
    filteredLogs.length - 1
  )
}

/** Page up/down in logs */
export function pageLogSelection(ctx: AppContext, direction: "up" | "down"): void {
  const { visibleCount } = getLogsLayoutMetrics(ctx)
  const delta = direction === "up" ? -visibleCount : visibleCount
  moveLogSelection(ctx, delta)
}

/** Toggle a log source filter */
export function toggleLogSource(ctx: AppContext, source: LogSource): void {
  const { appState } = ctx.state
  if (appState.logsSourceFilter.has(source)) {
    // Don't allow removing all filters
    if (appState.logsSourceFilter.size > 1) {
      appState.logsSourceFilter.delete(source)
    }
  } else {
    appState.logsSourceFilter.add(source)
  }
}

/** Enable tail mode (auto-scroll to latest) */
export function enableTailMode(ctx: AppContext): void {
  ctx.state.appState.logsTailMode = true
}

/** Set the active deployment for logs */
export function setActiveDeployment(ctx: AppContext, deploymentId: string | null): void {
  const { appState } = ctx.state
  appState.logsActiveDeploymentId = deploymentId
  appState.logsSelectedIndex = 0
  appState.logsWindowStart = 0
  appState.logsTailMode = true
}

/** Get list of available deployments for selection */
export function getDeploymentList(ctx: AppContext): Array<{ id: string; label: string }> {
  const { snapshot } = ctx.state
  return Array.from(snapshot.deployments.values()).map((d) => ({
    id: d.id,
    label: `${d.localApiPath} (${d.status})`,
  }))
}
