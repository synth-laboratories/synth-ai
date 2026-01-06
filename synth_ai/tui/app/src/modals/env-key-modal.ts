/**
 * Environment key scanner modal controller.
 */
import type { AppContext } from "../context"
import { scanEnvKeys } from "../utils/env"
import { clamp, type ModalController } from "./base"

export function createEnvKeyModal(ctx: AppContext): ModalController & {
  open: () => Promise<void>
  move: (delta: number) => void
  select: () => Promise<void>
  rescan: () => Promise<void>
} {
  const { ui, renderer } = ctx
  const { appState, backendKeys, backendKeySources, config } = ctx.state

  function toggle(visible: boolean): void {
    ui.envKeyModalVisible = visible
    ui.envKeyModalBox.visible = visible
    ui.envKeyModalTitle.visible = visible
    ui.envKeyModalHelp.visible = visible
    ui.envKeyModalListText.visible = visible
    ui.envKeyModalInfoText.visible = visible
    if (!visible && appState.activePane === "jobs") {
      ui.jobsSelect.focus()
    }
    renderer.requestRender()
  }

  function renderList(): void {
    if (appState.envKeyScanInProgress) {
      ui.envKeyModalListText.content = "Scanning..."
      ui.envKeyModalInfoText.content = ""
      renderer.requestRender()
      return
    }

    if (appState.envKeyError) {
      ui.envKeyModalListText.content = `Error: ${appState.envKeyError}`
      ui.envKeyModalInfoText.content = ""
      renderer.requestRender()
      return
    }

    if (!appState.envKeyOptions.length) {
      ui.envKeyModalListText.content = "No API keys found in .env files"
      ui.envKeyModalInfoText.content = ""
      renderer.requestRender()
      return
    }

    const max = Math.max(0, appState.envKeyOptions.length - 1)
    appState.envKeyCursor = clamp(appState.envKeyCursor, 0, max)
    const start = clamp(appState.envKeyWindowStart, 0, Math.max(0, max))
    const end = Math.min(appState.envKeyOptions.length, start + config.envKeyVisibleCount)

    const lines: string[] = []
    for (let idx = start; idx < end; idx++) {
      const option = appState.envKeyOptions[idx]
      const cursor = idx === appState.envKeyCursor ? ">" : " "
      const preview = option.key ? `${option.key.slice(0, 8)}...` : "(empty)"
      lines.push(`${cursor} ${preview}`)
    }
    ui.envKeyModalListText.content = lines.join("\n")

    const selected = appState.envKeyOptions[appState.envKeyCursor]
    if (selected) {
      const sources = selected.sources.slice(0, 2).join(", ")
      const suffix = selected.sources.length > 2 ? ` +${selected.sources.length - 2}` : ""
      ui.envKeyModalInfoText.content = `Source: ${sources}${suffix}\nVars: ${selected.varNames.join(", ")}`
    } else {
      ui.envKeyModalInfoText.content = ""
    }

    renderer.requestRender()
  }

  function move(delta: number): void {
    const max = Math.max(0, appState.envKeyOptions.length - 1)
    appState.envKeyCursor = clamp(appState.envKeyCursor + delta, 0, max)
    if (appState.envKeyCursor < appState.envKeyWindowStart) {
      appState.envKeyWindowStart = appState.envKeyCursor
    } else if (appState.envKeyCursor >= appState.envKeyWindowStart + config.envKeyVisibleCount) {
      appState.envKeyWindowStart = appState.envKeyCursor - config.envKeyVisibleCount + 1
    }
    renderList()
  }

  async function rescan(): Promise<void> {
    appState.envKeyScanInProgress = true
    appState.envKeyError = null
    renderList()

    try {
      appState.envKeyOptions = await scanEnvKeys(config.envKeyScanRoot)
      appState.envKeyCursor = 0
      appState.envKeyWindowStart = 0
    } catch (err: any) {
      appState.envKeyError = err?.message || "Scan failed"
    } finally {
      appState.envKeyScanInProgress = false
      renderList()
    }
  }

  async function open(): Promise<void> {
    toggle(true)
    await rescan()
  }

  async function select(): Promise<void> {
    const selected = appState.envKeyOptions[appState.envKeyCursor]
    if (!selected) return

    backendKeys[appState.currentBackend] = selected.key
    backendKeySources[appState.currentBackend] = {
      sourcePath: selected.sources[0] || null,
      varName: selected.varNames[0] || null,
    }
    toggle(false)

    const { persistSettings } = await import("../persistence/settings")
    await persistSettings({
      settingsFilePath: config.settingsFilePath,
      getCurrentBackend: () => appState.currentBackend,
      getBackendKey: (id) => backendKeys[id],
      getBackendKeySource: (id) => backendKeySources[id],
    })

    ctx.state.snapshot.status = "API key loaded from env file"
    ctx.render()
  }

  function handleKey(key: any): boolean {
    if (!ui.envKeyModalVisible) return false

    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    if (key.name === "return" || key.name === "enter") {
      void select()
      return true
    }
    if (key.name === "up" || key.name === "k") {
      move(-1)
      return true
    }
    if (key.name === "down" || key.name === "j") {
      move(1)
      return true
    }
    if (key.name === "r") {
      void rescan()
      return true
    }
    if (key.name === "m") {
      toggle(false)
      // Parent will open key modal
      return true
    }
    return true
  }

  return {
    get isVisible() {
      return ui.envKeyModalVisible
    },
    toggle,
    open,
    move,
    select,
    rescan,
    handleKey,
  }
}

