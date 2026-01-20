/**
 * Event detail modal controller.
 */
import type { AppContext } from "../context"
import { getFilteredEvents, formatEventData } from "../formatters"
import { createModalUI, clamp, wrapModalText, type ModalController } from "./base"
import { focusManager } from "../focus"

export function createEventModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  updateContent: () => void
} {
  const { renderer } = ctx
  const { appState, snapshot } = ctx.state

  const modal = createModalUI(renderer, {
    id: "event-modal",
    width: 80,
    height: 16,
    borderColor: "#60a5fa",
    titleColor: "#60a5fa",
    zIndex: 6,
  })

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "event-modal",
        handleKey,
      })
      modal.center()
    } else {
      focusManager.pop("event-modal")
      modal.setContent("")
    }
    modal.setVisible(visible)
  }

  function updateContent(): void {
    if (!modal.visible) return

    const filtered = getFilteredEvents(snapshot.events, appState.eventFilter)
    const event = filtered[appState.selectedEventIndex]
    if (!event) {
      modal.setContent("(no event)")
      return
    }

    const raw = event.message ?? formatEventData(event.data) ?? "(no data)"
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, (typeof process.stdout?.rows === "number" ? process.stdout.rows : 40) - 12)

    appState.eventModalOffset = clamp(appState.eventModalOffset, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(appState.eventModalOffset, appState.eventModalOffset + maxLines)

    modal.setTitle(`Event ${event.seq} - ${event.type}`)
    modal.setContent(visible.join("\n"))
    modal.setHint(
      wrapped.length > maxLines
        ? `[${appState.eventModalOffset + 1}-${appState.eventModalOffset + visible.length}/${wrapped.length}] j/k scroll | q close`
        : "q close"
    )
  }

  function move(delta: number): void {
    appState.eventModalOffset = Math.max(0, appState.eventModalOffset + delta)
    updateContent()
  }

  function open(): void {
    const filtered = getFilteredEvents(snapshot.events, appState.eventFilter)
    if (!filtered.length) return

    appState.eventModalOffset = 0
    modal.center()
    toggle(true)
    updateContent()
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
    handleKey,
  }

  return controller
}
