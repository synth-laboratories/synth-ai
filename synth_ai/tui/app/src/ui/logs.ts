/**
 * Log file listing + navigation helpers for logs pane.
 */
import { TextRenderable } from "@opentui/core"
import type { AppContext } from "../context"
import * as fs from "fs"
import * as path from "path"
import * as os from "os"

export type LogFileInfo = {
  path: string
  name: string
  mtimeMs: number
  size: number
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

function getLogsDirectory(): string {
  return path.join(os.homedir(), ".synth-ai", "tui", "logs")
}

export function listLogFiles(): LogFileInfo[] {
  const logsDir = getLogsDirectory()
  try {
    const entries = fs.readdirSync(logsDir)
    const files = entries
      .map((name) => {
        const fullPath = path.join(logsDir, name)
        const stat = fs.statSync(fullPath)
        if (!stat.isFile()) return null
        return {
          path: fullPath,
          name,
          mtimeMs: stat.mtimeMs,
          size: stat.size,
        }
      })
      .filter((file): file is LogFileInfo => Boolean(file))
    return files.sort((a, b) => b.mtimeMs - a.mtimeMs)
  } catch {
    return []
  }
}

function formatLogFileLabel(name: string): string {
  let type = "log"
  if (name.includes("_deploy_")) {
    type = "deploy"
  } else if (name.includes("_serve_")) {
    type = "serve"
  }

  const match = name.match(/(\d{4}_\d{2}_\d{2})_([0-9]{2}[:\-][0-9]{2}[:\-][0-9]{2})/)
  if (match) {
    const date = match[1]
    const time = match[2].replace(/-/g, ":")
    return `${date} ${time} ${type}`
  }

  return `${type} ${name}`
}

function formatLogFileRow(file: LogFileInfo, maxWidth: number): string {
  const label = formatLogFileLabel(file.name)
  const maxLen = Math.max(4, maxWidth - 2)
  return label.length > maxLen ? label.slice(0, maxLen - 3) + "..." : label
}

export function getSelectedLogFile(ctx: AppContext): LogFileInfo | null {
  const files = listLogFiles()
  const idx = ctx.state.appState.logsSelectedIndex
  return files[idx] ?? null
}

/** Render the logs pane */
export function renderLogs(ctx: AppContext): void {
  const { ui, renderer } = ctx
  const { appState } = ctx.state

  const files = listLogFiles()
  if (!files.length) {
    ui.logsContent.visible = false
    ui.logsEmptyText.visible = true
    ui.logsEmptyText.content = `No log files found.\n\n${getLogsDirectory()}`
    return
  }

  ui.logsEmptyText.visible = false
  ui.logsContent.visible = true

  const { visibleCount } = getLogsLayoutMetrics(ctx)
  const total = files.length

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

  const visibleFiles = files.slice(
    appState.logsWindowStart,
    appState.logsWindowStart + visibleCount
  )

  for (const entry of ui.logEntries) {
    ui.logsContent.remove(entry.text.id)
  }
  ui.logEntries = []

  const termWidth = typeof process.stdout?.columns === "number" ? process.stdout.columns : 80
  const maxWidth = termWidth - 4

  visibleFiles.forEach((file, index) => {
    const globalIndex = appState.logsWindowStart + index
    const isSelected = globalIndex === appState.logsSelectedIndex

    const content = formatLogFileRow(file, maxWidth)
    const text = new TextRenderable(renderer, {
      id: `log-entry-${index}`,
      content,
      fg: isSelected ? "#ffffff" : "#e2e8f0",
      bg: isSelected ? "#1e293b" : undefined,
    })

    ui.logsContent.add(text)
    ui.logEntries.push({ text })
  })

  const position = total > visibleCount
    ? ` [${appState.logsWindowStart + 1}-${Math.min(appState.logsWindowStart + visibleCount, total)}/${total}]`
    : ""
  ui.logsBox.title = `Logs (files)${position}`
}

/** Move log selection up or down */
export function moveLogSelection(ctx: AppContext, delta: number): void {
  const files = listLogFiles()
  if (!files.length) return

  const { appState } = ctx.state
  appState.logsSelectedIndex = clamp(
    appState.logsSelectedIndex + delta,
    0,
    files.length - 1
  )
}

/** Page up/down in logs */
export function pageLogSelection(ctx: AppContext, direction: "up" | "down"): void {
  const { visibleCount } = getLogsLayoutMetrics(ctx)
  const delta = direction === "up" ? -visibleCount : visibleCount
  moveLogSelection(ctx, delta)
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
