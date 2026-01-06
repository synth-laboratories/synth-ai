import { createModalUI, type ModalController } from "./base"
import type { AppContext } from "../context"

export function createCreateJobModal(ctx: AppContext): ModalController & { open: () => void } {
  const { renderer } = ctx
  const modal = createModalUI(renderer, {
    id: "create-job-modal",
    width: 60,
    height: 12,
    borderColor: "#10b981",
    titleColor: "#10b981",
    zIndex: 10,
  })

  function toggle(visible: boolean): void {
    modal.setVisible(visible)
  }

  function open(): void {
    modal.center()
    modal.setTitle("Create New Job")
    modal.setContent("(Coming soon)")
    modal.setHint("q close")
    toggle(true)
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false
    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true
  }

  return { get isVisible() { return modal.visible }, toggle, open, handleKey }
}
