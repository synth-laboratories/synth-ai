/**
 * Results (best snapshot) detail modal controller.
 */
import type { AppContext } from "../context"
import { formatResultsExpanded } from "../formatters"
import { copyToClipboard } from "../utils/clipboard"
import { centerModal, clamp, wrapModalText, type ModalController } from "./base"

export function createResultsModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  updateContent: () => void
  copyPrompt: () => Promise<void>
} {
  const { ui, renderer } = ctx
  const { appState, snapshot } = ctx.state

  function toggle(visible: boolean): void {
    ui.resultsModalVisible = visible
    ui.resultsModalBox.visible = visible
    ui.resultsModalTitle.visible = visible
    ui.resultsModalText.visible = visible
    ui.resultsModalHint.visible = visible
    if (!visible) {
      ui.resultsModalText.content = ""
    }
    renderer.requestRender()
  }

  function updateContent(): void {
    if (!ui.resultsModalVisible) return

    const raw = formatResultsExpanded(snapshot)
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, (typeof process.stdout?.rows === "number" ? process.stdout.rows : 40) - 12)

    appState.resultsModalOffset = clamp(appState.resultsModalOffset, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(appState.resultsModalOffset, appState.resultsModalOffset + maxLines)

    ui.resultsModalTitle.content = "Results - Best Snapshot"
    ui.resultsModalText.content = visible.join("\n")
    ui.resultsModalHint.content =
      wrapped.length > maxLines
        ? `[${appState.resultsModalOffset + 1}-${appState.resultsModalOffset + visible.length}/${wrapped.length}] j/k scroll | y copy | q close`
        : "y copy | q close"

    renderer.requestRender()
  }

  function move(delta: number): void {
    appState.resultsModalOffset = Math.max(0, appState.resultsModalOffset + delta)
    updateContent()
  }

  function open(): void {
    appState.resultsModalOffset = 0
    toggle(true)
    updateContent()
  }

  async function copyPrompt(): Promise<void> {
    const text = formatResultsExpanded(snapshot)
    if (text) {
      await copyToClipboard(text)
      snapshot.status = "Results copied to clipboard"
      ctx.render()
    }
  }

  function handleKey(key: any): boolean {
    if (!ui.resultsModalVisible) return false

    if (key.name === "up" || key.name === "k") {
      move(-1)
      return true
    }
    if (key.name === "down" || key.name === "j") {
      move(1)
      return true
    }
    if (key.name === "y") {
      void copyPrompt()
      return true
    }
    if (key.name === "return" || key.name === "enter" || key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true
  }

  return {
    get isVisible() {
      return ui.resultsModalVisible
    },
    toggle,
    open,
    move,
    updateContent,
    copyPrompt,
    handleKey,
  }
}

