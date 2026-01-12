/**
 * Settings (backend selection) modal controller.
 * Adapted for nightly's focusManager and createModalUI patterns.
 */
import type { AppContext } from "../context"
import type { BackendConfig } from "../types"
import { createModalUI, clamp, type ModalController, type ModalUI } from "./base"
import { focusManager } from "../focus"
import {
  appState,
  backendConfigs,
  frontendKeys,
  frontendKeySources,
  getKeyForBackend,
  getFrontendUrl,
} from "../state/app-state"

// Type declaration for Node.js process (available at runtime)
declare const process: {
  env: Record<string, string | undefined>
}

export type SettingsModalController = ModalController & {
  open: () => void
  move: (delta: number) => void
  select: () => Promise<void>
  openKeyModal: () => void
}

export function createSettingsModal(
  ctx: AppContext,
  deps?: {
    onOpenKeyModal?: () => void
    onBackendSwitch?: () => Promise<void>
  },
): SettingsModalController {
  const { renderer } = ctx
  const { config } = ctx.state

  // Create modal UI using the primitive
  const modal: ModalUI = createModalUI(renderer, {
    id: "settings-modal",
    width: 64,
    height: 14,
    borderColor: "#38bdf8",
    titleColor: "#38bdf8",
    zIndex: 10,
  })

  // Set initial content
  modal.setTitle("Settings - Backend")
  modal.setHint("j/k navigate  Enter select  Shift+K keys  q close")

  function buildSettingsOptions(): BackendConfig[] {
    return [backendConfigs.prod, backendConfigs.dev, backendConfigs.local]
  }

  function renderList(): void {
    const lines: string[] = []
    for (let idx = 0; idx < appState.settingsOptions.length; idx++) {
      const opt = appState.settingsOptions[idx]
      const active = appState.currentBackend === opt.id
      const cursor = idx === appState.settingsCursor ? ">" : " "
      lines.push(`${cursor} [${active ? "x" : " "}] ${opt.label} (${opt.id})`)
    }

    const selected = appState.settingsOptions[appState.settingsCursor]
    if (selected) {
      // Keys are stored by frontend URL, so dev and local share the same key
      // Show actual key for this frontend URL - no fallback to avoid confusion
      const key = getKeyForBackend(selected.id)
      const keyPreview = key.trim() ? `...${key.slice(-8)}` : "(no key)"
      const frontendUrl = getFrontendUrl(selected.id)

      lines.push("")
      lines.push(`Backend: ${selected.baseUrl}`)
      lines.push(`Frontend: ${frontendUrl}`)
      lines.push(`Key: ${keyPreview}`)
    }

    modal.setContent(lines.join("\n"))
    renderer.requestRender()
  }

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "settings-modal",
        handleKey,
      })
      modal.center()
      appState.settingsOptions = buildSettingsOptions()
      appState.settingsCursor = Math.max(
        0,
        appState.settingsOptions.findIndex((opt) => opt.id === appState.currentBackend),
      )
      renderList()
    } else {
      focusManager.pop("settings-modal")
    }
    modal.setVisible(visible)
  }

  function move(delta: number): void {
    const max = Math.max(0, appState.settingsOptions.length - 1)
    appState.settingsCursor = clamp(appState.settingsCursor + delta, 0, max)
    renderList()
  }

  async function select(): Promise<void> {
    const selected = appState.settingsOptions[appState.settingsCursor]
    if (!selected) return

    appState.currentBackend = selected.id

    // Update process.env so nightly's URL resolution picks it up
    // Remove /api suffix for the env var (it gets added by the API client)
    const baseUrl = selected.baseUrl.replace(/\/api$/, "")
    process.env.SYNTH_BACKEND_URL = baseUrl
    process.env.SYNTH_API_KEY = getKeyForBackend(selected.id) || ""

    toggle(false)
    ctx.state.snapshot.status = `Switching to ${selected.label}...`
    ctx.render()

    // Persist settings
    const { persistSettings } = await import("../persistence/settings")
    await persistSettings({
      settingsFilePath: config.settingsFilePath,
      getCurrentBackend: () => appState.currentBackend,
      getFrontendKey: (id) => frontendKeys[id],
      getFrontendKeySource: (id) => frontendKeySources[id],
    })

    // Trigger refresh after backend switch
    if (deps?.onBackendSwitch) {
      await deps.onBackendSwitch()
    }
  }

  function open(): void {
    toggle(true)
  }

  function openKeyModal(): void {
    toggle(false)
    deps?.onOpenKeyModal?.()
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
      void select()
      return true
    }
    if (key.name === "k" && key.shift) {
      openKeyModal()
      return true
    }
    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true // consume all keys when modal is open
  }

  return {
    get isVisible() {
      return modal.visible
    },
    toggle,
    open,
    move,
    select,
    openKeyModal,
    handleKey,
  }
}
