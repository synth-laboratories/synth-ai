/**
 * Profile modal controller - displays user and organization info.
 */
import type { AppContext } from "../context"
import { createModalUI, type ModalController, type ModalUI } from "./base"
import { focusManager } from "../focus"

export function createProfileModal(ctx: AppContext): ModalController & {
  open: () => void
} {
  const { renderer } = ctx
  const { snapshot } = ctx.state

  // Create modal UI using the primitive
  const modal: ModalUI = createModalUI(renderer, {
    id: "profile-modal",
    width: 72,
    height: 15,
    borderColor: "#818cf8",
    titleColor: "#818cf8",
    zIndex: 10,
  })

  // Set initial content
  modal.setTitle("Profile")
  modal.setHint("q close")

  function updateContent(): void {
    const org = snapshot.orgId || "-"
    const user = snapshot.userId || "-"
    const apiKey = process.env.SYNTH_API_KEY || "-"
    modal.setContent(`Organization:\n${org}\n\nUser:\n${user}\n\nAPI Key:\n${apiKey}`)
  }

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "profile-modal",
        handleKey,
      })
      modal.center()
      updateContent()
    } else {
      focusManager.pop("profile-modal")
    }
    modal.setVisible(visible)
  }

  function open(): void {
    toggle(true)
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false

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
    handleKey,
  }

  return controller
}
