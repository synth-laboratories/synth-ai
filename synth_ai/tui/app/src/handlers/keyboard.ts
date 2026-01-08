/**
 * Central keyboard handler - routes keypresses to the correct modal or app action.
 */
import type { AppContext } from "../context"
import type { LoginModalController } from "../login_modal"
import { setActivePane, setPrincipalPane } from "../ui/panes"
import { moveEventSelection, toggleSelectedEventExpanded } from "../ui/events"
import {
  scrollOpenCode,
  sendOpenCodeMessage,
  handleOpenCodeInput,
  handleOpenCodeBackspace,
  renderOpenCodePane,
} from "../ui/opencode"
import { refreshJobs, selectJob, cancelSelected, fetchArtifacts, fetchMetrics } from "../api/jobs"
import { refreshEvents } from "../api/events"
import { refreshIdentity } from "../api/identity"
import { getFilteredJobs } from "../selectors/jobs"

export type ModalControllers = {
  login: LoginModalController
  event: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  results: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  config: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  settings: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  filter: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  jobFilter: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  key: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  envKey: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => Promise<void> }
  snapshot: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  promptBrowser?: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  taskApps: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  usage: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => Promise<void> }
  sessions: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => Promise<void> }
}

export function createKeyboardHandler(
  ctx: AppContext,
  modals: ModalControllers,
): (key: any) => void {
  const { renderer, ui } = ctx
  const { appState, snapshot } = ctx.state

  return function handleKeypress(key: any): void {
    // Ctrl+C always quits
    if (key.ctrl && key.name === "c") {
      renderer.stop()
      renderer.destroy()
      process.exit(0)
    }

    // q/escape closes modals or quits
    // In OpenCode view, only escape is handled here (q is typed)
    const isQuitKey = key.name === "q" || key.name === "escape"
    const shouldHandleQuit = isQuitKey && !(key.name === "q" && appState.principalPane === "opencode")
    if (shouldHandleQuit) {
      if (modals.login.isVisible) {
        modals.login.toggle(false)
        return
      }
      if (modals.key.isVisible) {
        modals.key.handleKey(key)
        return
      }
      if (modals.envKey.isVisible) {
        modals.envKey.handleKey(key)
        return
      }
      if (modals.jobFilter.isVisible) {
        modals.jobFilter.handleKey(key)
        return
      }
      if (modals.settings.isVisible) {
        modals.settings.handleKey(key)
        return
      }
      if (modals.event.isVisible) {
        modals.event.handleKey(key)
        return
      }
      if (modals.results.isVisible) {
        modals.results.handleKey(key)
        return
      }
      if (modals.config.isVisible) {
        modals.config.handleKey(key)
        return
      }
      if (modals.filter.isVisible) {
        modals.filter.handleKey(key)
        return
      }
      if (modals.promptBrowser?.isVisible) {
        modals.promptBrowser.handleKey(key)
        return
      }
      if (modals.snapshot.isVisible) {
        modals.snapshot.handleKey(key)
        return
      }
      if (modals.taskApps.isVisible) {
        modals.taskApps.handleKey(key)
        return
      }
      if (modals.usage.isVisible) {
        modals.usage.handleKey(key)
        return
      }
      if (modals.sessions.isVisible) {
        modals.sessions.handleKey(key)
        return
      }
      // No modal open
      // In OpenCode view: escape goes back to jobs
      if (appState.principalPane === "opencode" && key.name === "escape") {
        setPrincipalPane(ctx, "jobs")
        return
      }
      // Quit
      renderer.stop()
      renderer.destroy()
      process.exit(0)
    }

    // Login modal captures all keys
    if (modals.login.isVisible) {
      if (key.name === "return" || key.name === "enter") {
        void modals.login.startAuth()
      }
      return
    }

    // Route to active modal
    if (modals.key.isVisible) {
      modals.key.handleKey(key)
      return
    }
    if (modals.envKey.isVisible) {
      modals.envKey.handleKey(key)
      return
    }
    if (modals.event.isVisible) {
      modals.event.handleKey(key)
      return
    }
    if (modals.results.isVisible) {
      modals.results.handleKey(key)
      return
    }
    if (modals.config.isVisible) {
      modals.config.handleKey(key)
      return
    }
    if (modals.promptBrowser?.isVisible) {
      modals.promptBrowser.handleKey(key)
      return
    }
    if (modals.settings.isVisible) {
      modals.settings.handleKey(key)
      return
    }
    if (modals.filter.isVisible) {
      modals.filter.handleKey(key)
      return
    }
    if (modals.jobFilter.isVisible) {
      modals.jobFilter.handleKey(key)
      return
    }
    if (modals.snapshot.isVisible) {
      modals.snapshot.handleKey(key)
      return
    }
    if (modals.taskApps.isVisible) {
      modals.taskApps.handleKey(key)
      return
    }
    if (modals.usage.isVisible) {
      modals.usage.handleKey(key)
      return
    }
    if (modals.sessions.isVisible) {
      modals.sessions.handleKey(key)
      return
    }

    // Global shortcuts that work in both views
    // Principal pane toggle (g = opencode view) - only when NOT typing
    if (key.name === "g" && !key.shift && !key.ctrl && appState.principalPane === "jobs") {
      setPrincipalPane(ctx, "opencode")
      return
    }

    // Sessions modal (o = opencode sessions) - only when NOT typing
    if (key.name === "o" && !key.shift && !key.ctrl && appState.principalPane === "jobs") {
      void modals.sessions.open()
      return
    }

    // OpenCode pane - typing mode by default
    // Use Escape to exit to jobs view, Ctrl+G for sessions
    if (appState.principalPane === "opencode") {
      // Escape exits to jobs view
      // (already handled above in q/escape section, but this is a reminder)

      // Ctrl+O opens sessions modal from opencode view
      if (key.name === "o" && key.ctrl) {
        void modals.sessions.open()
        return
      }

      // Arrow keys scroll the message history
      if (key.name === "up") {
        scrollOpenCode(ctx, -3)
        return
      }
      if (key.name === "down") {
        scrollOpenCode(ctx, 3)
        return
      }

      // Enter sends the message
      if (key.name === "return" || key.name === "enter") {
        void sendOpenCodeMessage(ctx)
        return
      }

      // Backspace deletes characters
      if (key.name === "backspace") {
        handleOpenCodeBackspace(ctx)
        return
      }

      // All other printable characters go to input
      if (key.sequence && !key.ctrl && !key.meta && key.sequence.length === 1) {
        handleOpenCodeInput(ctx, key.sequence)
        return
      }

      // Swallow other keys in opencode mode
      return
    }

    // Jobs view shortcuts below this point
    if (key.name === "tab") {
      setActivePane(ctx, appState.activePane === "jobs" ? "events" : "jobs")
      return
    }
    if (key.name === "e") {
      setActivePane(ctx, "events")
      return
    }
    if (key.name === "b") {
      setActivePane(ctx, "jobs")
      return
    }
    if (key.name === "g") {
      setPrincipalPane(ctx, "opencode")
      return
    }
    if (key.name === "o") {
      void modals.sessions.open()
      return
    }
    if (key.name === "l" && !key.shift) {
      modals.login.toggle(true)
      return
    }
    if (key.name === "l" && key.shift) {
      // Logout
      ctx.state.backendKeys[appState.currentBackend] = ""
      snapshot.status = "Logged out"
      ctx.render()
      return
    }
    if (key.name === "t") {
      modals.settings.open()
      return
    }
    if (key.name === "d") {
      void modals.usage.open()
      return
    }

    // Jobs view shortcuts (only when in jobs principal pane)
    if (key.name === "r") {
      void refreshJobs(ctx).then(() => ctx.render())
      return
    }
    if (key.name === "f") {
      modals.filter.open()
      return
    }
    if (key.name === "i") {
      modals.config.open()
      return
    }
    if (key.name === "p") {
      modals.results.open()
      return
    }
    if (key.name === "j" && key.shift) {
      modals.jobFilter.open()
      return
    }
    if (key.name === "s") {
      modals.snapshot.open()
      return
    }
    if (key.name === "u") {
      modals.taskApps.open()
      return
    }
    if (key.name === "c") {
      void cancelSelected(ctx).then(() => ctx.render())
      return
    }
    if (key.name === "a") {
      void fetchArtifacts(ctx).then(() => ctx.render())
      return
    }
    if (key.name === "m") {
      void fetchMetrics(ctx).then(() => ctx.render())
      return
    }

    // Jobs/Events pane-specific navigation
    if (appState.activePane === "events") {
      if (key.name === "up" || key.name === "k") {
        moveEventSelection(ctx, -1)
        ctx.render()
        return
      }
      if (key.name === "down" || key.name === "j") {
        moveEventSelection(ctx, 1)
        ctx.render()
        return
      }
      if (key.name === "return" || key.name === "enter") {
        modals.event.open()
        return
      }
      if (key.name === "x") {
        toggleSelectedEventExpanded(ctx)
        ctx.render()
        return
      }
    }

    // Jobs pane - let the select widget handle j/k navigation
  }
}

export function createPasteHandler(ctx: AppContext, keyModal: { isVisible: boolean }): (key: any) => void {
  const { ui, renderer } = ctx

  return function handlePaste(key: any): void {
    if (!keyModal.isVisible) return
    const seq = typeof key?.sequence === "string" ? key.sequence : ""
    if (!seq) return
    const cleaned = seq
      .replace("\u001b[200~", "")
      .replace("\u001b[201~", "")
      .replace(/\s+/g, "")
    if (!cleaned) return
    ui.keyModalInput.value = (ui.keyModalInput.value || "") + cleaned
    renderer.requestRender()
  }
}

