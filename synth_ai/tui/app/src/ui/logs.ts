/**
 * Log file listing + content viewer for logs pane.
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

/** Get layout metrics for logs content pane */
export function getLogsContentMetrics(): { visibleLines: number; maxWidth: number } {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 80
  // Reserve space for header, tabs, status, footer, box borders, and left panel
  const visibleLines = Math.max(1, rows - 12)
  const maxWidth = Math.max(20, cols - 42) // 36 for left panel + some padding
  return { visibleLines, maxWidth }
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

export function getSelectedLogFile(ctx: AppContext): LogFileInfo | null {
  const files = listLogFiles()
  const idx = ctx.state.appState.logsSelectedIndex
  return files[idx] ?? null
}

// Cache for log file content to avoid re-reading unchanged files
let cachedLogPath: string | null = null
let cachedLogContent: string[] | null = null
let cachedLogMtime: number = 0
let cachedLogMaxWidth: number = 0

/** Read and wrap file content for display (with caching) */
function readAndWrapContent(filePath: string, maxWidth: number): string[] {
  try {
    // Check if we can use cached content
    const stat = fs.statSync(filePath)
    if (
      filePath === cachedLogPath &&
      stat.mtimeMs === cachedLogMtime &&
      maxWidth === cachedLogMaxWidth &&
      cachedLogContent
    ) {
      return cachedLogContent
    }

    // Read and wrap content
    const content = fs.readFileSync(filePath, "utf-8")
    const lines: string[] = []
    for (const line of content.split("\n")) {
      if (line.length <= maxWidth) {
        lines.push(line)
      } else {
        // Wrap long lines
        for (let i = 0; i < line.length; i += maxWidth) {
          lines.push(line.slice(i, i + maxWidth))
        }
      }
    }

    // Update cache
    cachedLogPath = filePath
    cachedLogContent = lines
    cachedLogMtime = stat.mtimeMs
    cachedLogMaxWidth = maxWidth

    return lines
  } catch (err) {
    // Clear cache on error
    cachedLogPath = null
    cachedLogContent = null
    return [`Failed to read file: ${err}`]
  }
}

/** Render the logs content pane (shows selected file content) */
export function renderLogs(ctx: AppContext): void {
  const { ui, renderer } = ctx
  const { appState } = ctx.state

  const selectedFile = getSelectedLogFile(ctx)
  if (!selectedFile) {
    ui.logsContent.visible = false
    ui.logsEmptyText.visible = true
    ui.logsEmptyText.content = "No log file selected"
    ui.logsBox.title = "Log Content"
    return
  }

  ui.logsEmptyText.visible = false
  ui.logsContent.visible = true

  const { visibleLines, maxWidth } = getLogsContentMetrics()
  const wrappedLines = readAndWrapContent(selectedFile.path, maxWidth)
  const totalLines = wrappedLines.length

  // Handle tail mode - auto-scroll to bottom
  const maxOffset = Math.max(0, totalLines - visibleLines)
  if (appState.logsContentTailMode) {
    appState.logsContentOffset = maxOffset
  } else {
    appState.logsContentOffset = clamp(appState.logsContentOffset, 0, maxOffset)
  }

  // Get visible lines
  const visibleContent = wrappedLines.slice(
    appState.logsContentOffset,
    appState.logsContentOffset + visibleLines
  )

  // Clear existing entries
  for (const entry of ui.logEntries) {
    ui.logsContent.remove(entry.text.id)
  }
  ui.logEntries = []

  // Render content lines
  visibleContent.forEach((line, index) => {
    const text = new TextRenderable(renderer, {
      id: `log-line-${index}`,
      content: line || " ", // Empty lines need content to render
      fg: "#e2e8f0",
    })
    ui.logsContent.add(text)
    ui.logEntries.push({ text })
  })

  // Update title with position info
  const tailLabel = appState.logsContentTailMode ? " [TAIL]" : ""
  const position = totalLines > visibleLines
    ? ` [${appState.logsContentOffset + 1}-${Math.min(appState.logsContentOffset + visibleLines, totalLines)}/${totalLines}]`
    : ""
  ui.logsBox.title = `Log: ${path.basename(selectedFile.path)}${position}${tailLabel}`
}

/** Scroll log content up or down */
export function scrollLogContent(ctx: AppContext, delta: number): void {
  const { appState } = ctx.state
  appState.logsContentTailMode = false
  appState.logsContentOffset = Math.max(0, appState.logsContentOffset + delta)
}

/** Page up/down in log content */
export function pageLogContent(ctx: AppContext, direction: "up" | "down"): void {
  const { visibleLines } = getLogsContentMetrics()
  const delta = direction === "up" ? -visibleLines : visibleLines
  scrollLogContent(ctx, delta)
}

/** Toggle tail mode for log content */
export function toggleLogContentTailMode(ctx: AppContext): void {
  const { appState } = ctx.state
  appState.logsContentTailMode = !appState.logsContentTailMode
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
