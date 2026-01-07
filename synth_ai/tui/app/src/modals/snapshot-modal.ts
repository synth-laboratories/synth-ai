/**
 * Snapshot ID input modal controller.
 */
import type { AppContext } from "../context"
import { blurForModal, restoreFocusFromModal } from "../ui/panes"
import type { ModalController } from "./base"

export function createSnapshotModal(ctx: AppContext): ModalController & {
  open: () => void
  apply: (snapshotId: string) => Promise<void>
} {
  const { ui, renderer } = ctx
  const { appState, snapshot } = ctx.state

  function toggle(visible: boolean): void {
    ui.modalVisible = visible
    ui.modalBox.visible = visible
    ui.modalLabel.visible = visible
    ui.modalInput.visible = visible
    if (visible) {
      blurForModal(ctx)
      ui.modalInput.value = ""
      ui.modalInput.focus()
    } else {
      restoreFocusFromModal(ctx)
    }
    renderer.requestRender()
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
      await apiGet(`/prompt-learning/online/jobs/${job.job_id}/snapshots/${trimmed}`)
      snapshot.status = `Snapshot ${trimmed} fetched`
    } catch (err: any) {
      snapshot.lastError = err?.message || "Snapshot fetch failed"
    }
    ctx.render()
  }

  function handleKey(key: any): boolean {
    if (!ui.modalVisible) return false

    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    // Input is handled by InputRenderable directly
    return false
  }

  return {
    get isVisible() {
      return ui.modalVisible
    },
    toggle,
    open,
    apply,
    handleKey,
  }
}

