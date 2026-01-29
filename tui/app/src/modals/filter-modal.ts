/**
 * Event filter modal controller.
 */
import { InputRenderableEvents } from "@opentui/core"
import type { AppContext } from "../context"
import { createModalUI, type ModalController } from "./base"
import { focusManager } from "../focus"

export function createFilterModal(ctx: AppContext): ModalController & {
  open: () => void
} {
  const { renderer } = ctx
  const { appState } = ctx.state

  const modal = createModalUI(renderer, {
    id: "filter-modal",
    width: 52,
    height: 5,
    borderColor: "#60a5fa",
    titleColor: "#60a5fa",
    zIndex: 5,
    input: {
      label: "Event filter:",
      placeholder: "Type to filter events",
      width: 46,
    },
  })

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "filter-modal",
        handleKey,
      })
      modal.center()
      if (modal.input) {
        modal.input.value = appState.eventFilter
        modal.input.focus()
      }
    } else {
      focusManager.pop("filter-modal")
    }
    modal.setVisible(visible)
  }

  function open(): void {
    toggle(true)
  }

  function apply(value: string): void {
    appState.eventFilter = value.trim()
    toggle(false)
    ctx.render()
  }

  // Wire up input events
  if (modal.input) {
    modal.input.on(InputRenderableEvents.CHANGE, (value: string) => {
      apply(value)
    })
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false

    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    // Block all keys to prevent global shortcuts, InputRenderable handles input
    return true
  }

  const controller = {
    get isVisible() {
      return modal.visible
    },
    toggle,
    open,
    handleKey,
  }

  return controller
}
