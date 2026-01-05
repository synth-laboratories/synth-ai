/**
 * Settings (backend selection) modal controller.
 */
import type { AppContext } from "../context"
import type { BackendConfig, BackendId } from "../types"
import { clamp, type ModalController } from "./base"

export function createSettingsModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  select: () => Promise<void>
  openKeyModal: () => void
  openEnvKeyModal: () => void
} {
  const { ui, renderer } = ctx
  const { appState, backendConfigs, backendKeys } = ctx.state

  function buildSettingsOptions(): BackendConfig[] {
    return [backendConfigs.prod, backendConfigs.dev, backendConfigs.local]
  }

  function toggle(visible: boolean): void {
    ui.settingsModalVisible = visible
    ui.settingsBox.visible = visible
    ui.settingsTitle.visible = visible
    ui.settingsHelp.visible = visible
    ui.settingsListText.visible = visible
    ui.settingsInfoText.visible = visible
    if (visible) {
      appState.settingsOptions = buildSettingsOptions()
      appState.settingsCursor = Math.max(
        0,
        appState.settingsOptions.findIndex((opt) => opt.id === appState.currentBackend),
      )
      ui.jobsSelect.blur()
      renderList()
    } else if (appState.activePane === "jobs") {
      ui.jobsSelect.focus()
    }
    renderer.requestRender()
  }

  function renderList(): void {
    const lines: string[] = []
    for (let idx = 0; idx < appState.settingsOptions.length; idx++) {
      const opt = appState.settingsOptions[idx]
      const active = appState.currentBackend === opt.id
      const cursor = idx === appState.settingsCursor ? ">" : " "
      lines.push(`${cursor} [${active ? "x" : " "}] ${opt.label} (${opt.id})`)
    }
    ui.settingsListText.content = lines.join("\n")

    const selected = appState.settingsOptions[appState.settingsCursor]
    if (selected) {
      const key = backendKeys[selected.id]
      const keyPreview = key ? `${key.slice(0, 5)}...` : "(no key)"
      ui.settingsInfoText.content = `URL: ${selected.baseUrl}\nKey: ${keyPreview}`
    } else {
      ui.settingsInfoText.content = ""
    }
    renderer.requestRender()
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
    toggle(false)
    ctx.render()

    // Persist and reload
    const { persistSettings } = await import("../persistence/settings")
    await persistSettings({
      settingsFilePath: ctx.state.config.settingsFilePath,
      getCurrentBackend: () => appState.currentBackend,
      getBackendKey: (id) => backendKeys[id],
      getBackendKeySource: (id) => ctx.state.backendKeySources[id],
    })
  }

  function open(): void {
    toggle(true)
  }

  function openKeyModal(): void {
    toggle(false)
    // This will be handled by the parent that wires modals together
  }

  function openEnvKeyModal(): void {
    toggle(false)
    // This will be handled by the parent that wires modals together
  }

  function handleKey(key: any): boolean {
    if (!ui.settingsModalVisible) return false

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
    if (key.name === "e" && key.shift) {
      openEnvKeyModal()
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
      return ui.settingsModalVisible
    },
    toggle,
    open,
    move,
    select,
    openKeyModal,
    openEnvKeyModal,
    handleKey,
  }
}

