/**
 * Main app orchestrator - creates renderer, wires controllers/handlers, bootstraps polling.
 */
import { createCliRenderer, SelectRenderableEvents, InputRenderableEvents } from "@opentui/core"

import { createAppContext, type AppContext } from "./context"
import { buildLayout } from "./components/layout"
import { renderApp } from "./ui/render"

import { createLoginModal } from "./login_modal"
import {
  createEventModal,
  createResultsModal,
  createConfigModal,
  createSettingsModal,
  createFilterModal,
  createJobFilterModal,
  createKeyModal,
  createEnvKeyModal,
  createSnapshotModal,
  createTaskAppsModal,
} from "./modals"

import { createKeyboardHandler, createPasteHandler } from "./handlers/keyboard"
import { loadPersistedSettings, persistSettings } from "./persistence/settings"
import { normalizeBackendId } from "./state/app-state"
import { refreshJobs, selectJob } from "./api/jobs"
import { refreshEvents } from "./api/events"
import { refreshIdentity, refreshHealth } from "./api/identity"
import { refreshTunnels, refreshTunnelHealth } from "./api/tunnels"
import { getActiveApiKey } from "./api/client"

// Type declaration for Node.js process (available at runtime)
declare const process: {
  env: Record<string, string | undefined>
}

export async function runApp(): Promise<void> {
  // Create renderer
  const renderer = await createCliRenderer({
    useConsole: false,
    useAlternateScreen: true,
    openConsoleOnError: false,
    backgroundColor: "#0b1120",
  })

  // Build layout with a placeholder footer (will be updated by render)
  const ui = buildLayout(renderer, () => "")

  // Create context
  let ctx: AppContext

  function render(): void {
    renderApp(ctx)
  }

  ctx = createAppContext({
    renderer,
    ui,
    render,
  })

  // Load persisted settings
  await loadPersistedSettings({
    settingsFilePath: ctx.state.config.settingsFilePath,
    normalizeBackendId,
    setCurrentBackend: (id) => {
      ctx.state.appState.currentBackend = id
    },
    setBackendKey: (id, key) => {
      ctx.state.backendKeys[id] = key
    },
    setBackendKeySource: (id, source) => {
      ctx.state.backendKeySources[id] = source
    },
  })
  
  // Ensure local backend has SYNTH_API_KEY fallback if still empty
  const synthApiKey = process.env.SYNTH_API_KEY || process.env.SYNTH_TUI_API_KEY_LOCAL || ""
  if (!ctx.state.backendKeys.local?.trim() && synthApiKey) {
    ctx.state.backendKeys.local = synthApiKey
  }

  // Create modal controllers
  const loginModal = createLoginModal({
    ui,
    renderer,
    getCurrentBackend: () => ctx.state.appState.currentBackend,
    getBackendConfig: () => ctx.state.backendConfigs[ctx.state.appState.currentBackend],
    getBackendKeys: () => ctx.state.backendKeys,
    setBackendKey: (backend, key, source) => {
      ctx.state.backendKeys[backend] = key
      ctx.state.backendKeySources[backend] = source
    },
    persistSettings: async () => {
      await persistSettings({
        settingsFilePath: ctx.state.config.settingsFilePath,
        getCurrentBackend: () => ctx.state.appState.currentBackend,
        getBackendKey: (id) => ctx.state.backendKeys[id],
        getBackendKeySource: (id) => ctx.state.backendKeySources[id],
      })
    },
    bootstrap: async () => {
      await bootstrap()
    },
    getSnapshot: () => ctx.state.snapshot,
    renderSnapshot: render,
    getActivePane: () => ctx.state.appState.activePane,
  })

  const eventModal = createEventModal(ctx)
  const resultsModal = createResultsModal(ctx)
  const configModal = createConfigModal(ctx)
  const settingsModal = createSettingsModal(ctx)
  const filterModal = createFilterModal(ctx)
  const jobFilterModal = createJobFilterModal(ctx)
  const keyModal = createKeyModal(ctx)
  const envKeyModal = createEnvKeyModal(ctx)
  const snapshotModal = createSnapshotModal(ctx)
  const taskAppsModal = createTaskAppsModal(ctx)

  const modals = {
    login: loginModal,
    event: eventModal,
    results: resultsModal,
    config: configModal,
    settings: settingsModal,
    filter: filterModal,
    jobFilter: jobFilterModal,
    key: keyModal,
    envKey: envKeyModal,
    snapshot: snapshotModal,
    taskApps: taskAppsModal,
  }

  // Create keyboard handler
  const handleKeypress = createKeyboardHandler(ctx, modals)
  const handlePaste = createPasteHandler(ctx, keyModal)

  // Wire up event listeners
  renderer.keyInput.on("keypress", handleKeypress)
  renderer.keyInput.on("paste", handlePaste)

  // Jobs select widget
  ui.jobsSelect.on(SelectRenderableEvents.SELECTION_CHANGED, (_idx: number, option: any) => {
    if (!option?.value) return
    if (ctx.state.snapshot.selectedJob?.job_id !== option.value) {
      void selectJob(ctx, option.value).then(() => render())
    }
  })

  // Snapshot modal input
  ui.modalInput.on(InputRenderableEvents.CHANGE, (value: string) => {
    if (!value.trim()) {
      snapshotModal.toggle(false)
      return
    }
    void snapshotModal.apply(value.trim())
  })
  ui.modalInput.on(InputRenderableEvents.ENTER, (value: string) => {
    if (!value.trim()) {
      snapshotModal.toggle(false)
      return
    }
    void snapshotModal.apply(value.trim())
  })

  // Filter input
  ui.filterInput.on(InputRenderableEvents.CHANGE, (value: string) => {
    ctx.state.appState.eventFilter = value.trim()
    filterModal.toggle(false)
    render()
  })

  // Key modal input
  ui.keyModalInput.on(InputRenderableEvents.ENTER, (value: string) => {
    void keyModal.apply(value)
  })

  // Start renderer
  renderer.start()
  ui.jobsSelect.focus()
  render()

  // Bootstrap
  if (!getActiveApiKey()) {
    ctx.state.snapshot.lastError = `Missing API key for ${ctx.state.backendConfigs[ctx.state.appState.currentBackend].label}`
    ctx.state.snapshot.status = "Sign in required"
    render()
    loginModal.toggle(true)
  } else {
    bootstrap().catch((err) => {
      ctx.state.snapshot.lastError = err?.message || "Bootstrap failed"
      ctx.state.snapshot.status = "Startup error"
      render()
    })
  }

  // Bootstrap function
  async function bootstrap(): Promise<void> {
    void refreshHealth(ctx)
    await refreshIdentity(ctx)
    await refreshJobs(ctx)

    // Load tunnels (task apps) and check their health
    await refreshTunnels(ctx)
    void refreshTunnelHealth(ctx).then(() => render())

    const { initialJobId } = ctx.state.config
    if (initialJobId) {
      await selectJob(ctx, initialJobId)
    } else if (ctx.state.snapshot.jobs.length > 0) {
      await selectJob(ctx, ctx.state.snapshot.jobs[0].job_id)
    }

    scheduleJobsPoll()
    scheduleEventsPoll()
    setInterval(() => void refreshHealth(ctx), 30_000)
    setInterval(() => void refreshIdentity(ctx).then(() => render()), 60_000)
    // Refresh tunnels every 30 seconds, health checks every 15 seconds
    setInterval(() => void refreshTunnels(ctx).then(() => render()), 30_000)
    setInterval(() => void refreshTunnelHealth(ctx).then(() => render()), 15_000)
    render()
  }

  // Polling
  function scheduleJobsPoll(): void {
    const { pollingState, config } = ctx.state
    if (pollingState.jobsTimer) clearTimeout(pollingState.jobsTimer)
    pollingState.jobsTimer = setTimeout(pollJobs, pollingState.jobsPollMs)
  }

  async function pollJobs(): Promise<void> {
    const { pollingState, config } = ctx.state
    if (pollingState.jobsInFlight) {
      scheduleJobsPoll()
      return
    }
    pollingState.jobsInFlight = true
    const ok = await refreshJobs(ctx)
    pollingState.jobsInFlight = false
    render()

    if (ok) {
      pollingState.jobsPollMs = Math.max(1, config.refreshInterval) * 1000
    } else {
      pollingState.jobsPollMs = Math.min(
        pollingState.jobsPollMs * 2,
        Math.max(1, config.maxRefreshInterval) * 1000,
      )
    }
    scheduleJobsPoll()
  }

  function scheduleEventsPoll(): void {
    const { pollingState } = ctx.state
    if (pollingState.eventsTimer) clearTimeout(pollingState.eventsTimer)
    pollingState.eventsTimer = setTimeout(pollEvents, pollingState.eventsPollMs)
  }

  async function pollEvents(): Promise<void> {
    const { pollingState, config } = ctx.state
    if (pollingState.eventsInFlight || !ctx.state.snapshot.selectedJob) {
      scheduleEventsPoll()
      return
    }
    pollingState.eventsInFlight = true
    const ok = await refreshEvents(ctx)
    pollingState.eventsInFlight = false
    render()

    if (ok) {
      pollingState.eventsPollMs = Math.max(0.5, config.eventInterval) * 1000
    } else {
      pollingState.eventsPollMs = Math.min(
        pollingState.eventsPollMs * 2,
        Math.max(1, config.maxEventInterval) * 1000,
      )
    }
    scheduleEventsPoll()
  }
}

