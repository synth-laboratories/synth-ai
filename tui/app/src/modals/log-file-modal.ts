/**
 * Log file viewer modal controller.
 */
import type { AppContext } from "../context"
import { copyToClipboard } from "../utils/clipboard"
import { createModalUI, clamp, wrapModalText, type ModalController } from "./base"
import { focusManager } from "../focus"
import { registerCleanup, unregisterCleanup } from "../lifecycle"
import * as fs from "fs"
import * as path from "path"

export function createLogFileModal(ctx: AppContext): ModalController & {
  open: (filePath: string) => void
  move: (delta: number) => void
  updateContent: () => void
  copyContent: () => Promise<void>
} {
  const { renderer } = ctx
  const { appState, snapshot } = ctx.state

  const modalHeight = 24
  const modal = createModalUI(renderer, {
    id: "log-file-modal",
    width: 100,
    height: modalHeight,
    borderColor: "#38bdf8",
    titleColor: "#38bdf8",
    zIndex: 9,
  })

  let currentFilePath: string | null = null
  let refreshTimer: ReturnType<typeof setInterval> | null = null
  const cleanupName = "log-file-modal-refresh"

  function startAutoRefresh(): void {
    if (refreshTimer) return
    refreshTimer = setInterval(() => {
      if (modal.visible) updateContent()
    }, 1000)
    registerCleanup(cleanupName, () => {
      if (refreshTimer) {
        clearInterval(refreshTimer)
        refreshTimer = null
      }
    })
  }

  function stopAutoRefresh(): void {
    if (refreshTimer) {
      clearInterval(refreshTimer)
      refreshTimer = null
    }
    unregisterCleanup(cleanupName)
  }

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "log-file-modal",
        handleKey,
      })
      modal.center()
      startAutoRefresh()
    } else {
      focusManager.pop("log-file-modal")
      modal.setContent("")
      stopAutoRefresh()
    }
    modal.setVisible(visible)
  }

  function readFileContent(filePath: string): string {
    try {
      return fs.readFileSync(filePath, "utf-8")
    } catch (err) {
      return `Failed to read file: ${err}`
    }
  }

  function updateContent(): void {
    if (!modal.visible || !currentFilePath) return

    const raw = readFileContent(currentFilePath)
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, modalHeight - 5)

    const maxOffset = Math.max(0, wrapped.length - maxLines)
    if (appState.logsModalTail) {
      appState.logsModalOffset = maxOffset
    } else {
      appState.logsModalOffset = clamp(appState.logsModalOffset, 0, maxOffset)
    }
    const visible = wrapped.slice(appState.logsModalOffset, appState.logsModalOffset + maxLines)

    modal.setTitle(`Log File: ${path.basename(currentFilePath)}`)
    modal.setContent(visible.join("\n"))
    const tailLabel = appState.logsModalTail ? " [TAIL]" : ""
    modal.setHint(
      wrapped.length > maxLines
        ? `[${appState.logsModalOffset + 1}-${appState.logsModalOffset + visible.length}/${wrapped.length}] j/k scroll | t tail${tailLabel} | y copy | q close`
        : `t tail${tailLabel} | y copy | q close`
    )
  }

  function move(delta: number): void {
    appState.logsModalTail = false
    appState.logsModalOffset = Math.max(0, appState.logsModalOffset + delta)
    updateContent()
  }

  function open(filePath: string): void {
    currentFilePath = filePath
    appState.logsModalTail = true
    toggle(true)
    updateContent()
  }

  async function copyContent(): Promise<void> {
    if (!currentFilePath) return
    const raw = readFileContent(currentFilePath)
    await copyToClipboard(raw)
    snapshot.status = `Copied: ${path.basename(currentFilePath)}`
    ctx.render()
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
      void copyContent()
      return true
    }
    if (key.name === "t") {
      appState.logsModalTail = true
      updateContent()
      return true
    }
    if (key.name === "return" || key.name === "enter" || key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true // consume all keys when modal is open
  }

  return {
    get isVisible() {
      return modal.visible
    },
    toggle,
    open,
    move,
    updateContent,
    copyContent,
    handleKey,
  }
}
