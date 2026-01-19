/**
 * Job configuration/metadata modal controller.
 */
import type { AppContext } from "../context"
import { createModalUI, clamp, wrapModalText, type ModalController } from "./base"
import { focusManager } from "../focus"

export function createConfigModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  updateContent: () => void
} {
  const { renderer } = ctx
  const { appState, snapshot } = ctx.state

  const modal = createModalUI(renderer, {
    id: "config-modal",
    width: 100,
    height: 24,
    borderColor: "#f59e0b",
    titleColor: "#f59e0b",
    zIndex: 8,
  })

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "config-modal",
        handleKey,
      })
      modal.center()
    } else {
      focusManager.pop("config-modal")
      modal.setContent("")
    }
    modal.setVisible(visible)
  }

  function formatConfigMetadata(): string | null {
    const job = snapshot.selectedJob
    if (!job) return null

    const lines: string[] = []
    lines.push(`Job: ${job.job_id}`)
    lines.push(`Status: ${job.status}`)
    lines.push(`Type: ${job.job_type || "-"}`)
    lines.push(`Source: ${job.job_source || "unknown"}`)
    lines.push("")

    if (snapshot.lastError && snapshot.status?.includes("Error")) {
      lines.push("═══ Error Loading Metadata ═══")
      lines.push(snapshot.lastError)
      lines.push("")
      lines.push("The job details could not be loaded.")
      return lines.join("\n")
    }

    const meta: any = job.metadata
    if (!meta || Object.keys(meta).length === 0) {
      if (snapshot.status?.includes("Loading")) {
        lines.push("Loading job configuration...")
        lines.push("")
        lines.push("Modal will auto-update when loaded.")
      } else if (!job.job_type) {
        lines.push("Loading job configuration...")
        lines.push("")
        lines.push("Press 'i' again after job details finish loading.")
      } else {
        lines.push("No metadata available for this job.")
        lines.push("")
        lines.push(`(job_source: ${job.job_source}, job_type: ${job.job_type})`)
      }
      return lines.join("\n")
    }

    const desc = meta.request_metadata?.description || meta.description
    if (desc) {
      lines.push(`Description: ${desc}`)
      lines.push("")
    }

    const rawConfig =
      meta.prompt_initial_snapshot?.raw_config?.prompt_learning ||
      meta.config?.prompt_learning ||
      meta.job_config?.prompt_learning ||
      meta.prompt_learning ||
      meta.config ||
      meta.job_config ||
      null

    const optimizerConfig = meta.prompt_initial_snapshot?.optimizer_config || meta.optimizer_config || null

    const policy = rawConfig?.policy || optimizerConfig?.policy_config
    if (policy) {
      lines.push("═══ Model Configuration ═══")
      if (policy.model) lines.push(`  Model: ${policy.model}`)
      if (policy.provider) lines.push(`  Provider: ${policy.provider}`)
      if (policy.temperature != null) lines.push(`  Temperature: ${policy.temperature}`)
      if (policy.max_completion_tokens) lines.push(`  Max Tokens: ${policy.max_completion_tokens}`)
      lines.push("")
    }

    // Add more config sections as needed...
    try {
      const metaJson = JSON.stringify(meta, null, 2)
      if (metaJson.length < 2000) {
        lines.push("═══ Raw Metadata ═══")
        lines.push(metaJson)
      }
    } catch {
      // ignore
    }

    return lines.join("\n")
  }

  function updateContent(): void {
    if (!modal.visible) return

    const raw = formatConfigMetadata() || "(no metadata)"
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, (typeof process.stdout?.rows === "number" ? process.stdout.rows : 40) - 12)

    appState.configModalOffset = clamp(appState.configModalOffset, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(appState.configModalOffset, appState.configModalOffset + maxLines)

    modal.setTitle("Job Configuration")
    modal.setContent(visible.join("\n"))
    modal.setHint(
      wrapped.length > maxLines
        ? `[${appState.configModalOffset + 1}-${appState.configModalOffset + visible.length}/${wrapped.length}] j/k scroll | q close`
        : "q close"
    )
  }

  function move(delta: number): void {
    appState.configModalOffset = Math.max(0, appState.configModalOffset + delta)
    updateContent()
  }

  function open(): void {
    appState.configModalOffset = 0
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
    if (key.name === "return" || key.name === "enter" || key.name === "i" || key.name === "q" || key.name === "escape") {
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
