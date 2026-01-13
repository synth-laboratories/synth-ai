/**
 * Settings (mode selection) modal controller.
 * Adapted for nightly's focusManager and createModalUI patterns.
 */
import type { AppContext } from "../context"
import type { Mode } from "../types"
import { createModalUI, clamp, type ModalController, type ModalUI } from "./base"
import { focusManager } from "../focus"
import { appState, modeKeys, modeUrls, switchMode } from "../state/app-state"

const MODE_LABELS: Record<Mode, string> = {
  prod: "Prod",
  dev: "Dev",
  local: "Local",
}

const MODES: Mode[] = ["prod", "dev", "local"]

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
    onModeSwitch?: () => Promise<void>
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
  modal.setTitle("Settings - Mode")
  modal.setHint("j/k navigate  Enter select  Shift+K keys  q close")

  function renderList(): void {
    const lines: string[] = []
    for (let idx = 0; idx < MODES.length; idx++) {
      const mode = MODES[idx]
      const active = appState.currentMode === mode
      const cursor = idx === appState.settingsCursor ? ">" : " "
      lines.push(`${cursor} [${active ? "x" : " "}] ${MODE_LABELS[mode]} (${mode})`)
    }

    const selectedMode = MODES[appState.settingsCursor]
    if (selectedMode) {
      const urls = modeUrls[selectedMode]
      const key = modeKeys[selectedMode]
      const keyPreview = key.trim() ? `...${key.slice(-8)}` : "(no key)"
      lines.push("")
      lines.push(`Backend: ${urls.backendUrl || "(unset)"}`)
      lines.push(`Frontend: ${urls.frontendUrl || "(unset)"}`)
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
      appState.settingsOptions = [...MODES]
      appState.settingsCursor = Math.max(0, MODES.indexOf(appState.currentMode))
      renderList()
    } else {
      focusManager.pop("settings-modal")
    }
    modal.setVisible(visible)
  }

  function move(delta: number): void {
    const max = Math.max(0, MODES.length - 1)
    appState.settingsCursor = clamp(appState.settingsCursor + delta, 0, max)
    renderList()
  }

  async function select(): Promise<void> {
    const selectedMode = MODES[appState.settingsCursor]
    if (!selectedMode) return

    const urls = modeUrls[selectedMode]
    if (!urls.backendUrl || !urls.frontendUrl) {
      ctx.state.snapshot.status = `Missing URLs for ${MODE_LABELS[selectedMode]}.`
      ctx.render()
      return
    }

    // Switch mode (updates env vars and state)
    switchMode(selectedMode)

    toggle(false)
    ctx.state.snapshot.status = `Switching to ${MODE_LABELS[selectedMode]}...`
    ctx.render()

    // Persist settings
    const { persistSettings } = await import("../persistence/settings")
    await persistSettings({
      settingsFilePath: config.settingsFilePath,
      getCurrentMode: () => appState.currentMode,
      getModeKeys: () => modeKeys,
    })

    // Trigger refresh after mode switch
    if (deps?.onModeSwitch) {
      await deps.onModeSwitch()
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
