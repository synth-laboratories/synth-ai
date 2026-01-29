/**
 * Central keyboard handler - routes keypresses to the correct modal or app action.
 */
import type { AppContext } from "../context"
import type { LoginModalController } from "../login_modal"
import { shutdown } from "../lifecycle"
import { setActivePane, cycleActivePane, togglePrincipalPane } from "../ui/panes"
import { refreshJobs, cancelSelected, fetchArtifacts, fetchMetrics } from "../api/jobs"
import { focusManager } from "../focus"

export type ModalControllers = {
  login: LoginModalController
  filter: { open: () => void }
  jobFilter: { open: () => void }
  key: { open: () => void }
  settings: { open: () => void }
  snapshot: { open: () => void }
  profile: { open: () => void }
  urls: { open: () => void }
  usage: { open: () => Promise<void>; isVisible: boolean; handleKey?: (key: any) => boolean }
  event: { open: () => void }
  results: { open: () => void }
  config: { open: () => void }
  createJob: { open: () => void }
  taskApps: { open: () => void }
  logFile: { open: (filePath: string) => void }
  sessions: { open: () => Promise<void>; isVisible: boolean; handleKey?: (key: any) => boolean }
}

export function createKeyboardHandler(
  ctx: AppContext,
  modals: ModalControllers,
): (key: any) => void {
  return function handleKeypress(key: any): void {
    // Ctrl+C always quits.
    if (key.ctrl && key.name === "c") {
      void shutdown(0)
      return
    }

    if (modals.usage.isVisible) {
      modals.usage.handleKey?.(key)
      return
    }
    if (modals.sessions.isVisible) {
      modals.sessions.handleKey?.(key)
      return
    }

    if (ctx.state.appState.principalPane === "opencode") {
      if (key.ctrl && key.name === "x" && ctx.state.appState.openCodeAbort) {
        ctx.state.appState.openCodeAbort()
        return
      }
      if (key.name === "escape" && ctx.state.appState.openCodeAbort) {
        ctx.state.appState.openCodeAbort()
        return
      }
      if (key.name === "g" && key.shift) {
        togglePrincipalPane(ctx)
      }
      if (key.name === "o" && key.shift) {
        void modals.sessions.open()
      }
      return
    }

    // Route to focused item (modal or pane)
    if (focusManager.handleKey(key)) {
      return
    }

    // No modal consumed the key - q/escape quits
    if (key.name === "q" || key.name === "escape") {
      void shutdown(0)
      return
    }

    // Global shortcuts
    if (key.name === "tab") {
      cycleActivePane(ctx)
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
    if (key.name === "g" && !key.shift) {
      setActivePane(ctx, "logs")
      return
    }
    if (key.name === "g" && key.shift) {
      togglePrincipalPane(ctx)
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
    if (key.name === "v") {
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
    if (key.name === "t") {
      modals.settings.open()
      return
    }
    if (key.name === "d") {
      void modals.usage.open()
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
    if (key.name === "o" && key.shift) {
      void modals.sessions.open()
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

    // Pane-specific navigation is handled by focus system (see panes.ts)
    // Jobs pane uses the select widget's built-in navigation
  }
}
