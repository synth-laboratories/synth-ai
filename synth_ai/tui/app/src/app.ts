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
  createFilterModal,
  createJobFilterModal,
  createKeyModal,
  createSnapshotModal,
  createProfileModal,
  createUrlsModal,
  createTaskAppsModal,
} from "./modals"
import { createCreateJobModal } from "./modals/create-job-modal"

import { createKeyboardHandler, createPasteHandler } from "./handlers/keyboard"
import { refreshJobs, selectJob } from "./api/jobs"
import { refreshEvents } from "./api/events"
import { refreshIdentity, refreshHealth } from "./api/identity"
import { refreshTunnels, refreshTunnelHealth } from "./api/tunnels"
import { connectJobsStream, type JobStreamEvent } from "./api/jobs-stream"
import { isLoggedOutMarkerSet, loadSavedApiKey } from "./utils/logout-marker"
import { clearJobsTimer, pollingState } from "./state/polling"
import { registerRenderer, registerInterval, registerCleanup, installSignalHandlers } from "./lifecycle"

export async function runApp(): Promise<void> {
  // Create renderer
  const renderer = await createCliRenderer({
    useConsole: false,
    useAlternateScreen: true,
    openConsoleOnError: false,
    backgroundColor: "#0b1120",
  })

  // Register renderer for centralized shutdown and install signal handlers
  registerRenderer(renderer)
  installSignalHandlers()

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

  // Create modal controllers
  const loginModal = createLoginModal({
    renderer,
    bootstrap: async () => {
      await bootstrap()
    },
    getSnapshot: () => ctx.state.snapshot,
    renderSnapshot: render,
    getActivePane: () => ctx.state.appState.activePane,
    focusJobsSelect: () => ui.jobsSelect.focus(),
    blurJobsSelect: () => ui.jobsSelect.blur(),
  })

  const eventModal = createEventModal(ctx)
  const resultsModal = createResultsModal(ctx)
  const configModal = createConfigModal(ctx)
  const filterModal = createFilterModal(ctx)
  const jobFilterModal = createJobFilterModal(ctx)
  const keyModal = createKeyModal(ctx)
  const snapshotModal = createSnapshotModal(ctx)
  const profileModal = createProfileModal(ctx)
  const urlsModal = createUrlsModal(renderer)
  const createJobModal = createCreateJobModal(ctx)
  const taskAppsModal = createTaskAppsModal(ctx)

  const modals = {
    login: loginModal,
    event: eventModal,
    results: resultsModal,
    config: configModal,
    filter: filterModal,
    jobFilter: jobFilterModal,
    key: keyModal,
    snapshot: snapshotModal,
    profile: profileModal,
    urls: urlsModal,
    createJob: createJobModal,
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
  const loggedOutMarkerSet = isLoggedOutMarkerSet()

  if (loggedOutMarkerSet) {
    // User explicitly logged out previously - don't auto-login
    ctx.state.snapshot.status = "Sign in required"
    render()
    loginModal.toggle(true)
  } else {
    // Try to load saved API key if not already in env
    if (!process.env.SYNTH_API_KEY) {
      const savedKey = loadSavedApiKey()
      if (savedKey) {
        process.env.SYNTH_API_KEY = savedKey
      }
    }

    if (!process.env.SYNTH_API_KEY) {
      // No API key available
      ctx.state.snapshot.status = "Sign in required"
      render()
      loginModal.toggle(true)
    } else {
      // Try auto-login with API key
      tryAutoLogin().catch(() => {
        // Silent fail - show logged out state
        process.env.SYNTH_API_KEY = ""
        ctx.state.snapshot.status = "Sign in required"
        render()
        loginModal.toggle(true)
      })
    }
  }

  async function tryAutoLogin(): Promise<void> {
    // Validate API key by fetching identity
    await refreshIdentity(ctx)

    // If identity fetch failed (no userId), treat as auth failure
    if (!ctx.state.snapshot.userId) {
      throw new Error("Invalid API key")
    }

    // Key is valid - proceed with full bootstrap
    await bootstrap()
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

    // Start SSE connection for real-time job updates
    startJobsStream()

    // Events polling still needed (SSE is only for job list metadata)
    scheduleEventsPoll()
    registerInterval(setInterval(() => void refreshHealth(ctx), 30_000))
    registerInterval(setInterval(() => void refreshIdentity(ctx).then(() => render()), 60_000))
    // Refresh tunnels every 30 seconds, health checks every 15 seconds
    registerInterval(setInterval(() => void refreshTunnels(ctx).then(() => render()), 30_000))
    registerInterval(setInterval(() => void refreshTunnelHealth(ctx).then(() => render()), 15_000))

    // Register SSE disconnect for cleanup on shutdown
    registerCleanup("sse", () => pollingState.sseDisconnect?.())

    render()
  }

  // SSE stream for jobs list
  function startJobsStream(): void {
    const { pollingState } = ctx.state

    if (!process.env.SYNTH_API_KEY) {
      // No API key, fall back to polling
      scheduleJobsPoll()
      return
    }

    const stream = connectJobsStream(
      (event) => handleJobStreamEvent(event),
      (err) => handleJobStreamError(err),
      pollingState.lastSseSeq,
    )

    // SSE connected - stop job polling
    pollingState.sseConnected = true
    pollingState.sseDisconnect = stream.disconnect
    pollingState.sseReconnectDelay = 1000 // Reset backoff
    clearJobsTimer()
  }

  function handleJobStreamEvent(event: JobStreamEvent): void {
    const { snapshot, pollingState } = ctx.state

    // Track sequence for reconnection
    pollingState.lastSseSeq = event.seq

    // Remember currently selected job to restore selection after render
    const selectedJobId = snapshot.selectedJob?.job_id

    const idx = snapshot.jobs.findIndex((j) => j.job_id === event.job_id)

    if (event.type === "job.created" && idx === -1) {
      // Add new job to top of list
      snapshot.jobs.unshift({
        job_id: event.job_id,
        status: event.status,
        training_type: event.algorithm ?? null,
        created_at: event.created_at ?? null,
        started_at: event.started_at ?? null,
        finished_at: event.finished_at ?? null,
        best_score: null,
        best_snapshot_id: null,
        total_tokens: null,
        total_cost_usd: null,
        error: event.error ?? null,
        job_source: event.job_type === "prompt_learning" ? "prompt-learning" : "learning",
      })
    } else if (idx !== -1) {
      // Update existing job
      const job = snapshot.jobs[idx]
      job.status = event.status
      if (event.started_at) job.started_at = event.started_at
      if (event.finished_at) job.finished_at = event.finished_at
      if (event.error) job.error = event.error
    }

    render()

    // Restore selection to previously selected job (not the new one)
    if (selectedJobId) {
      const newIdx = snapshot.jobs.findIndex((j) => j.job_id === selectedJobId)
      if (newIdx !== -1) {
        ui.jobsSelect.setSelectedIndex(newIdx)
      }
    }
  }

  function handleJobStreamError(_err: Error): void {
    const { pollingState } = ctx.state

    // SSE disconnected
    pollingState.sseConnected = false
    pollingState.sseDisconnect = null

    // Resume polling as fallback
    scheduleJobsPoll()

    // Schedule reconnect with exponential backoff
    if (pollingState.sseReconnectTimer) {
      clearTimeout(pollingState.sseReconnectTimer)
    }
    pollingState.sseReconnectTimer = setTimeout(() => {
      pollingState.sseReconnectTimer = null
      startJobsStream()
    }, pollingState.sseReconnectDelay)

    // Exponential backoff: 1s, 2s, 4s, ... up to 30s
    pollingState.sseReconnectDelay = Math.min(pollingState.sseReconnectDelay * 2, 30_000)
  }

  // Polling
  function scheduleJobsPoll(): void {
    const { pollingState } = ctx.state
    // Don't schedule polling if SSE is connected
    if (pollingState.sseConnected) return
    if (pollingState.jobsTimer) clearTimeout(pollingState.jobsTimer)
    pollingState.jobsTimer = setTimeout(pollJobs, pollingState.jobsPollMs)
  }

  async function pollJobs(): Promise<void> {
    const { pollingState, config } = ctx.state
    // Skip polling if SSE is connected
    if (pollingState.sseConnected) return
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

