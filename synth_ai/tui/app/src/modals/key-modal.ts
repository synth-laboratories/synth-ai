/**
 * API key input modal controller.
 */
import type { AppContext } from "../context"
import { createModalUI, type ModalController } from "./base"
import { focusManager } from "../focus"

export function createKeyModal(ctx: AppContext): ModalController & {
  open: () => void
  apply: (value: string) => Promise<void>
  paste: () => void
} {
  const { renderer } = ctx

  const modal = createModalUI(renderer, {
    id: "key-modal",
    width: 70,
    height: 7,
    borderColor: "#7dd3fc",
    titleColor: "#7dd3fc",
    zIndex: 10,
    input: {
      label: "API Key:",
      placeholder: "",
      width: 62,
    },
  })

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "key-modal",
        handleKey,
      })
      modal.center()
      if (modal.input) {
        modal.input.value = ""
        modal.input.focus()
      }
      modal.setHint("Paste or type key | Enter to apply | q to cancel")
    } else {
      focusManager.pop("key-modal")
    }
    modal.setVisible(visible)
  }

  function open(): void {
    toggle(true)
  }

  async function apply(value: string): Promise<void> {
    const trimmed = value.trim()
    if (!trimmed) {
      toggle(false)
      return
    }

    process.env.SYNTH_API_KEY = trimmed
    toggle(false)

    ctx.state.snapshot.status = "API key updated"
    ctx.render()
  }

  function paste(): void {
    try {
      if (process.platform !== "darwin") return
      const result = require("child_process").spawnSync("pbpaste", [], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      })
      if (result.status !== 0) return
      const text = result.stdout ? String(result.stdout).replace(/\s+/g, "") : ""
      if (!text) return
      if (modal.input) {
        modal.input.value = (modal.input.value || "") + text
      }
    } catch {
      // ignore
    }
    renderer.requestRender()
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false

    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    if (key.name === "return" || key.name === "enter") {
      void apply(modal.input?.value || "")
      return true
    }
    if (key.name === "v" && (key.ctrl || key.meta)) {
      paste()
      return true
    }
    if (key.name === "backspace" || key.name === "delete") {
      if (modal.input) {
        const current = modal.input.value || ""
        modal.input.value = current.slice(0, Math.max(0, current.length - 1))
      }
      renderer.requestRender()
      return true
    }
    // Handle character input manually
    const seq = key.sequence || ""
    if (seq && !seq.startsWith("\u001b") && !key.ctrl && !key.meta) {
      if (modal.input) {
        modal.input.value = (modal.input.value || "") + seq
      }
      renderer.requestRender()
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
    apply,
    paste,
    handleKey,
  }

  return controller
}
