/**
 * Event filter modal controller.
 */
import type { AppContext } from "../context"
import type { ModalController } from "./base"

export function createFilterModal(ctx: AppContext): ModalController & {
  open: () => void
} {
  const { ui, renderer } = ctx
  const { appState } = ctx.state

  function toggle(visible: boolean): void {
    ui.filterModalVisible = visible
    ui.filterBox.visible = visible
    ui.filterLabel.visible = visible
    ui.filterInput.visible = visible
    if (visible) {
      ui.filterInput.value = appState.eventFilter
      ui.filterInput.focus()
    } else if (appState.activePane === "jobs") {
      ui.jobsSelect.focus()
    }
    renderer.requestRender()
  }

  function open(): void {
    toggle(true)
  }

  function apply(value: string): void {
    appState.eventFilter = value.trim()
    toggle(false)
    ctx.render()
  }

  function handleKey(key: any): boolean {
    if (!ui.filterModalVisible) return false

    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    // Input is handled by InputRenderable directly
    return false
  }

  // Hook up the input change event
  ui.filterInput.on?.("change", (value: string) => {
    apply(value)
  })

  return {
    get isVisible() {
      return ui.filterModalVisible
    },
    toggle,
    open,
    handleKey,
  }
}

