/**
 * Results (best snapshot) detail modal controller.
 */
import type { AppContext } from "../context"
import { formatResultsExpanded } from "../formatters"
import { copyToClipboard } from "../utils/clipboard"
import { createModalUI, clamp, wrapModalText, type ModalController } from "./base"
import { focusManager } from "../focus"

export function createResultsModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  updateContent: () => void
  copyPrompt: () => Promise<void>
} {
  const { renderer } = ctx
  const { appState, snapshot } = ctx.state

  const modal = createModalUI(renderer, {
    id: "results-modal",
    width: 100,
    height: 24,
    borderColor: "#22c55e",
    titleColor: "#22c55e",
    zIndex: 8,
  })

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "results-modal",
        handleKey,
      })
      modal.center()
    } else {
      focusManager.pop("results-modal")
      modal.setContent("")
    }
    modal.setVisible(visible)
  }

  function updateContent(): void {
    if (!modal.visible) return

    const raw = formatResultsExpanded(snapshot) ?? "No results available"
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    // Modal height is 24, minus 2 for borders, 1 for title, 2 for hint area = 19 usable lines
    const maxLines = 19

    appState.resultsModalOffset = clamp(appState.resultsModalOffset, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(appState.resultsModalOffset, appState.resultsModalOffset + maxLines)

    modal.setTitle("Results - Best Snapshot")
    modal.setContent(visible.join("\n"))
    modal.setHint(
      wrapped.length > maxLines
        ? `[${appState.resultsModalOffset + 1}-${appState.resultsModalOffset + visible.length}/${wrapped.length}] j/k scroll | y copy | q close`
        : "y copy | q close"
    )
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
      void copyPrompt()
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
    copyPrompt,
    handleKey,
  }

  return controller
}
