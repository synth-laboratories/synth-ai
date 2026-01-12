/**
 * Snapshot ID input modal controller.
 */
import { InputRenderableEvents } from "@opentui/core"
import type { AppContext } from "../context"
import { createModalUI, type ModalController } from "./base"
import { focusManager } from "../focus"
import { getAbortSignal } from "../lifecycle/shutdown"

export function createSnapshotModal(ctx: AppContext): ModalController & {
  open: () => void
  apply: (snapshotId: string) => Promise<void>
} {
  const { renderer } = ctx
  const { snapshot } = ctx.state

  const modal = createModalUI(renderer, {
    id: "snapshot-modal",
    width: 50,
    height: 5,
    borderColor: "#60a5fa",
    titleColor: "#60a5fa",
    zIndex: 5,
    input: {
      label: "Snapshot ID:",
      placeholder: "Enter snapshot id",
      width: 44,
    },
  })

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "snapshot-modal",
        handleKey,
      })
      modal.center()
      if (modal.input) {
        modal.input.value = ""
        modal.input.focus()
      }
    } else {
      focusManager.pop("snapshot-modal")
    }
    modal.setVisible(visible)
  }

  function open(): void {
    toggle(true)
  }

  async function apply(snapshotId: string): Promise<void> {
    const trimmed = snapshotId.trim()
    if (!trimmed) {
      toggle(false)
      return
    }

    const job = snapshot.selectedJob
    if (!job) {
      toggle(false)
      return
    }

    toggle(false)
    try {
      const { apiGet } = await import("../api/client")
      await apiGet(`/prompt-learning/online/jobs/${job.job_id}/snapshots/${trimmed}`, { signal: getAbortSignal() })
      snapshot.status = `Snapshot ${trimmed} fetched`
    } catch (err: any) {
      snapshot.lastError = err?.message || "Snapshot fetch failed"
    }
    ctx.render()
  }

  // Wire up input events
  if (modal.input) {
    modal.input.on(InputRenderableEvents.ENTER, (value: string) => {
      void apply(value)
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
    apply,
    handleKey,
  }

  return controller
}
