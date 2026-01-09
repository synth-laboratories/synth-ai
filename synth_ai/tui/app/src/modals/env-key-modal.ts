/**
 * Environment key scanner modal controller.
 * Adapted for nightly's focusManager and createModalUI patterns.
 */
import type { AppContext } from "../context"
import { createModalUI, clamp, type ModalController, type ModalUI } from "./base"
import { focusManager } from "../focus"
import { scanEnvKeys } from "../utils/env"
import {
  appState,
  frontendKeys,
  frontendKeySources,
  getFrontendUrlId,
} from "../state/app-state"

export type EnvKeyModalController = ModalController & {
  open: () => Promise<void>
  move: (delta: number) => void
  select: () => Promise<void>
  rescan: () => Promise<void>
}

export function createEnvKeyModal(ctx: AppContext): EnvKeyModalController {
  const { renderer } = ctx
  const { config } = ctx.state

  // Create modal UI using the primitive
  const modal: ModalUI = createModalUI(renderer, {
    id: "env-key-modal",
    width: 64,
    height: 16,
    borderColor: "#a78bfa",
    titleColor: "#a78bfa",
    zIndex: 11, // Above settings modal
  })

  // Set initial content
  modal.setTitle("Scan .env Files for API Keys")
  modal.setHint("j/k navigate  Enter select  r rescan  q close")

  function renderList(): void {
    if (appState.envKeyScanInProgress) {
      modal.setContent("Scanning...")
      renderer.requestRender()
      return
    }

    if (appState.envKeyError) {
      modal.setContent(`Error: ${appState.envKeyError}`)
      renderer.requestRender()
      return
    }

    if (!appState.envKeyOptions.length) {
      const lines = [
        "No API keys found in .env files",
        "",
        `Scanned: ${config.envKeyScanRoot}`,
        "",
        "Looking for vars:",
        "  SYNTH_API_KEY",
        "  SYNTH_TUI_API_KEY_PROD",
        "  SYNTH_TUI_API_KEY_DEV",
        "  SYNTH_TUI_API_KEY_LOCAL",
      ]
      modal.setContent(lines.join("\n"))
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

    const selected = appState.envKeyOptions[appState.envKeyCursor]
    if (selected) {
      const sources = selected.sources.slice(0, 2).join(", ")
      const suffix = selected.sources.length > 2 ? ` +${selected.sources.length - 2}` : ""
      lines.push("")
      lines.push(`Source: ${sources}${suffix}`)
      lines.push(`Vars: ${selected.varNames.join(", ")}`)
    }

    modal.setContent(lines.join("\n"))
    renderer.requestRender()
  }

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "env-key-modal",
        handleKey,
      })
      modal.center()
    } else {
      focusManager.pop("env-key-modal")
    }
    modal.setVisible(visible)
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
    if (!selected) {
      // Close modal when no keys available (pressing enter should dismiss)
      toggle(false)
      return
    }

    // Store by frontend URL (dev and local share the same key)
    const frontendUrlId = getFrontendUrlId(appState.currentBackend)
    frontendKeys[frontendUrlId] = selected.key
    frontendKeySources[frontendUrlId] = {
      sourcePath: selected.sources[0] || null,
      varName: selected.varNames[0] || null,
    }

    // Also update process.env for immediate use
    process.env.SYNTH_API_KEY = selected.key

    toggle(false)

    const { persistSettings } = await import("../persistence/settings")
    await persistSettings({
      settingsFilePath: config.settingsFilePath,
      getCurrentBackend: () => appState.currentBackend,
      getFrontendKey: (id) => frontendKeys[id],
      getFrontendKeySource: (id) => frontendKeySources[id],
    })

    ctx.state.snapshot.status = "API key loaded from env file"
    ctx.render()
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false

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
    rescan,
    handleKey,
  }
}
