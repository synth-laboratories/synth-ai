/**
 * Central keyboard handler - routes keypresses to the correct modal or app action.
 */
import type { AppContext } from "../context"
import type { LoginModalController } from "../login_modal"
import { shutdown } from "../lifecycle"
import { setActivePane } from "../ui/panes"
import { moveEventSelection, toggleSelectedEventExpanded } from "../ui/events"
import { refreshJobs, selectJob, cancelSelected, fetchArtifacts, fetchMetrics } from "../api/jobs"
import { refreshEvents } from "../api/events"
import { refreshIdentity } from "../api/identity"
import { getFilteredJobs } from "../selectors/jobs"

export type ModalControllers = {
  login: LoginModalController
  event: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  results: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  config: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  filter: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  jobFilter: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  key: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  snapshot: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  profile: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  urls: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  promptBrowser?: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  createJob: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
  taskApps: { isVisible: boolean; handleKey: (key: any) => boolean; open: () => void }
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
      void shutdown(0)
      return
    }

    // q/escape closes modals or quits
    if (key.name === "q" || key.name === "escape") {
      if (modals.login.isVisible) {
        modals.login.toggle(false)
        return
      }
      if (modals.key.isVisible) {
        modals.key.handleKey(key)
        return
      }
      if (modals.jobFilter.isVisible) {
        modals.jobFilter.handleKey(key)
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
      if (modals.profile.isVisible) {
        modals.profile.handleKey(key)
        return
      }
      if (modals.urls.isVisible) {
        modals.urls.handleKey(key)
        return
      }
      if (modals.createJob.isVisible) {
        modals.createJob.handleKey(key)
        return
      }
      if (modals.taskApps.isVisible) {
        modals.taskApps.handleKey(key)
        return
      }
      // No modal open - quit
      void shutdown(0)
      return
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
    if (modals.profile.isVisible) {
      modals.profile.handleKey(key)
      return
    }
    if (modals.urls.isVisible) {
      modals.urls.handleKey(key)
      return
    }
    if (modals.createJob.isVisible) {
      modals.createJob.handleKey(key)
      return
    }
    if (modals.taskApps.isVisible) {
      modals.taskApps.handleKey(key)
      return
    }

    // Global shortcuts
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
    if (key.name === "r") {
      void refreshJobs(ctx).then(() => ctx.render())
      return
    }
    if (key.name === "l") {
      void modals.login.logout()
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
      if (!process.env.SYNTH_API_KEY) return // do nothing if not logged in
      modals.profile.open()
      return
    }
    if (key.name === "o") {
      modals.results.open()
      return
    }
    if (key.name === "j" && key.shift) {
      modals.jobFilter.open()
      return
    }
    if (key.name === "s" && !key.shift) {
      modals.snapshot.open()
      return
    }
    if (key.name === "s" && key.shift) {
      modals.urls.open()
      return
    }
    if (key.name === "n") {
      modals.createJob.open()
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

    // Pane-specific navigation
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

