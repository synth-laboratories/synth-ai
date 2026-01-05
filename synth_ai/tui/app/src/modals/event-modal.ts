/**
 * Event detail modal controller.
 */
import type { AppContext } from "../context"
import { getFilteredEvents, formatEventData } from "../formatters"
import { centerModal, clamp, wrapModalText, type ModalController } from "./base"

export function createEventModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  updateContent: () => void
} {
  const { ui, renderer } = ctx
  const { appState, snapshot } = ctx.state

  function toggle(visible: boolean): void {
    ui.eventModalVisible = visible
    ui.eventModalBox.visible = visible
    ui.eventModalTitle.visible = visible
    ui.eventModalText.visible = visible
    ui.eventModalHint.visible = visible
    if (!visible) {
      ui.eventModalText.content = ""
    }
    renderer.requestRender()
  }

  function updateContent(): void {
    if (!ui.eventModalVisible) return

    const filtered = getFilteredEvents(snapshot.events, appState.eventFilter)
    const event = filtered[appState.selectedEventIndex]
    if (!event) {
      ui.eventModalText.content = "(no event)"
      renderer.requestRender()
      return
    }

    const raw = event.message ?? formatEventData(event.data) ?? "(no data)"
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, (typeof process.stdout?.rows === "number" ? process.stdout.rows : 40) - 12)

    appState.eventModalOffset = clamp(appState.eventModalOffset, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(appState.eventModalOffset, appState.eventModalOffset + maxLines)

    ui.eventModalTitle.content = `Event ${event.seq} - ${event.type}`
    ui.eventModalText.content = visible.join("\n")
    ui.eventModalHint.content =
      wrapped.length > maxLines
        ? `[${appState.eventModalOffset + 1}-${appState.eventModalOffset + visible.length}/${wrapped.length}] j/k scroll | q close`
        : "q close"

    renderer.requestRender()
  }

  function move(delta: number): void {
    appState.eventModalOffset = Math.max(0, appState.eventModalOffset + delta)
    updateContent()
  }

  function open(): void {
    const filtered = getFilteredEvents(snapshot.events, appState.eventFilter)
    if (!filtered.length) return

    appState.eventModalOffset = 0
    toggle(true)
    updateContent()
  }

  function handleKey(key: any): boolean {
    if (!ui.eventModalVisible) return false

    if (key.name === "up" || key.name === "k") {
      move(-1)
      return true
    }
    if (key.name === "down" || key.name === "j") {
      move(1)
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
      return ui.eventModalVisible
    },
    toggle,
    open,
    move,
    updateContent,
    handleKey,
  }
}

