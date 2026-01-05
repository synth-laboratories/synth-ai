/**
 * API key input modal controller.
 */
import type { AppContext } from "../context"
import type { ModalController } from "./base"

export function createKeyModal(ctx: AppContext): ModalController & {
  open: () => void
  apply: (value: string) => Promise<void>
  paste: () => void
} {
  const { ui, renderer } = ctx
  const { appState, backendKeys, backendKeySources, config } = ctx.state

  function toggle(visible: boolean): void {
    ui.keyModalVisible = visible
    ui.keyModalBox.visible = visible
    ui.keyModalLabel.visible = visible
    ui.keyModalInput.visible = visible
    ui.keyModalHelp.visible = visible
    if (visible) {
      ui.keyModalInput.value = ""
      ui.keyModalInput.focus()
      ui.keyModalLabel.content = `API Key for ${appState.keyModalBackend}:`
      ui.keyModalHelp.content = "Paste or type key | Enter to apply | q to cancel"
    } else if (appState.activePane === "jobs") {
      ui.jobsSelect.focus()
    }
    renderer.requestRender()
  }

  function open(): void {
    appState.keyModalBackend = appState.currentBackend
    toggle(true)
  }

  async function apply(value: string): Promise<void> {
    const trimmed = value.trim()
    if (!trimmed) {
      toggle(false)
      return
    }

    backendKeys[appState.keyModalBackend] = trimmed
    backendKeySources[appState.keyModalBackend] = {
      sourcePath: "manual-input",
      varName: null,
    }
    toggle(false)

    const { persistSettings } = await import("../persistence/settings")
    await persistSettings({
      settingsFilePath: config.settingsFilePath,
      getCurrentBackend: () => appState.currentBackend,
      getBackendKey: (id) => backendKeys[id],
      getBackendKeySource: (id) => backendKeySources[id],
    })

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
      ui.keyModalInput.value = (ui.keyModalInput.value || "") + text
    } catch {
      // ignore
    }
    renderer.requestRender()
  }

  function handleKey(key: any): boolean {
    if (!ui.keyModalVisible) return false

    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    if (key.name === "return" || key.name === "enter") {
      void apply(ui.keyModalInput.value || "")
      return true
    }
    if (key.name === "v" && (key.ctrl || key.meta)) {
      paste()
      return true
    }
    if (key.name === "backspace" || key.name === "delete") {
      const current = ui.keyModalInput.value || ""
      ui.keyModalInput.value = current.slice(0, Math.max(0, current.length - 1))
      renderer.requestRender()
      return true
    }
    // Handle character input
    const seq = key.sequence || ""
    if (seq && !seq.startsWith("\u001b") && !key.ctrl && !key.meta) {
      ui.keyModalInput.value = (ui.keyModalInput.value || "") + seq
      renderer.requestRender()
      return true
    }
    return true
  }

  return {
    get isVisible() {
      return ui.keyModalVisible
    },
    toggle,
    open,
    apply,
    paste,
    handleKey,
  }
}

