/**
 * Settings (backend selection) modal controller.
 * Allows switching between prod/dev/local backends.
 */
import type { AppContext } from "../context"
import type { BackendId } from "../types"
import { createModalUI, clamp, type ModalController, type ModalUI } from "./base"

export function createSettingsModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  select: () => void
} {
  const { renderer } = ctx
  const { appState, backendConfigs, backendKeys } = ctx.state

  const modal: ModalUI = createModalUI(renderer, {
    id: "settings-modal",
    width: 60,
    height: 18,
    borderColor: "#f59e0b",
    titleColor: "#f59e0b",
    zIndex: 10,
  })

  modal.setTitle("Settings - Backend")
  modal.setHint("j/k navigate | enter select | q close")

  const backendIds: BackendId[] = ["prod", "dev", "local"]

  function getKeyPreview(id: BackendId): string {
    let key = backendKeys[id]
    if (!key || !key.trim()) {
      // Check env fallback
      key = process.env.SYNTH_API_KEY || ""
    }
    return key && key.trim() ? `${key.slice(0, 8)}...` : "(no key)"
  }

  function updateContent(): void {
    const lines: string[] = []

    for (let idx = 0; idx < backendIds.length; idx++) {
      const id = backendIds[idx]
      const config = backendConfigs[id]
      const isActive = appState.currentBackend === id
      const isCursor = idx === appState.settingsCursor
      const cursor = isCursor ? ">" : " "
      const active = isActive ? "[x]" : "[ ]"
      const keyPreview = getKeyPreview(id)

      lines.push(`${cursor} ${active} ${config.label}`)
      lines.push(`     URL: ${config.baseUrl.replace(/\/api$/, "")}`)
      lines.push(`     Key: ${keyPreview}`)
      lines.push("")
    }

    modal.setContent(lines.join("\n"))
  }

  function toggle(visible: boolean): void {
    if (visible) {
      // Reset cursor to current backend
      appState.settingsCursor = Math.max(
        0,
        backendIds.findIndex((id) => id === appState.currentBackend),
      )
      modal.center()
      updateContent()
    }
    modal.setVisible(visible)
  }

  function open(): void {
    toggle(true)
  }

  function move(delta: number): void {
    const max = backendIds.length - 1
    appState.settingsCursor = clamp(appState.settingsCursor + delta, 0, max)
    updateContent()
  }

  function select(): void {
    const selectedId = backendIds[appState.settingsCursor]
    if (!selectedId) return

    appState.currentBackend = selectedId

    // Update process.env so the API client uses the new backend
    const config = backendConfigs[selectedId]
    process.env.SYNTH_BACKEND_URL = config.baseUrl.replace(/\/api$/, "")
    process.env.SYNTH_FRONTEND_URL = config.frontendUrl

    // Update API key if we have one for this backend
    const key = backendKeys[selectedId]
    if (key && key.trim()) {
      process.env.SYNTH_API_KEY = key
    }

    toggle(false)
    ctx.state.snapshot.status = `Switched to ${config.label}`
    ctx.render()
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
    if (key.name === "return" || key.name === "enter") {
      select()
      return true
    }
    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true
  }

  return {
    get isVisible() {
      return modal.visible
    },
    toggle,
    open,
    move,
    select,
    handleKey,
  }
}
