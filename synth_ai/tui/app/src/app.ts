/**
 * Main app orchestrator - creates renderer, wires controllers/handlers, bootstraps polling.
 */
import { createCliRenderer, SelectRenderableEvents } from "@opentui/core"

import { createAppContext, type AppContext } from "./context"
import { JobSource } from "./utils/job-types"
import { buildLayout } from "./components/layout"
import { renderApp } from "./ui/render"

import { createLoginModal } from "./login_modal"
import {
  createEventModal,
  createResultsModal,
  createConfigModal,
  createEnvKeyModal,
  createFilterModal,
  createJobFilterModal,
  createKeyModal,
  createLogFileModal,
  createSettingsModal,
  createSnapshotModal,
  createProfileModal,
  createUrlsModal,
  createUsageModal,
  createTaskAppsModal,
  createSessionsModal,
} from "./modals"
import { createCreateJobModal } from "./modals/create-job-modal"

import { createKeyboardHandler } from "./handlers/keyboard"
import { focusManager } from "./focus"
import { initPaneFocusables } from "./ui/panes"
import { refreshJobs, selectJob } from "./api/jobs"
import { refreshEvents } from "./api/events"
import { refreshIdentity, refreshHealth } from "./api/identity"
import { refreshTunnels, refreshTunnelHealth } from "./api/tunnels"
import { connectJobsStream, type JobStreamEvent } from "./api/jobs-stream"
import { connectJobDetailsStream, type JobDetailsStreamEvent, type JobDetailsStreamConnection } from "./api/job-details-stream"
import { isLoggedOutMarkerSet, loadSavedApiKey } from "./utils/logout-marker"
import { clearJobsTimer, pollingState, config, disconnectJobDetailsSse, clearJobDetailsSseTimer } from "./state/polling"
import type { JobResultRow } from "./types"
import { registerRenderer, registerInterval, registerTimeout, unregisterTimeout, registerCleanup, installSignalHandlers } from "./lifecycle"
import { loadPersistedSettings } from "./persistence/settings"
import { appState, normalizeBackendId, frontendKeys, frontendKeySources, getKeyForBackend, backendConfigs } from "./state/app-state"
import { startOpenCodeServer } from "./utils/opencode-server"

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

  // Register renderer for centralized shutdown and install signal handlers
  registerRenderer(renderer)
  installSignalHandlers()

  // Load persisted settings (backend selection and API keys)
  await loadPersistedSettings({
    settingsFilePath: config.settingsFilePath,
    normalizeBackendId,
    setCurrentBackend: (id) => { appState.currentBackend = id },
    setFrontendKey: (id, key) => { frontendKeys[id] = key },
    setFrontendKeySource: (id, source) => { frontendKeySources[id] = source },
  })

  // Update process.env with loaded settings
  const currentConfig = backendConfigs[appState.currentBackend]
  process.env.SYNTH_BACKEND_URL = currentConfig.baseUrl.replace(/\/api$/, "")
  process.env.SYNTH_API_KEY = getKeyForBackend(appState.currentBackend) || process.env.SYNTH_API_KEY || ""

  // Build layout with a placeholder footer (will be updated by render)
  const ui = buildLayout(renderer, () => "")

  // Create context
  let ctx: AppContext

  function render(): void {
    try {
      renderApp(ctx)
    } catch {
      // Ignore render errors - don't crash the app
    }
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
  })

  const eventModal = createEventModal(ctx)
  const resultsModal = createResultsModal(ctx)
  const configModal = createConfigModal(ctx)
  const filterModal = createFilterModal(ctx)
  const jobFilterModal = createJobFilterModal(ctx)
  const keyModal = createKeyModal(ctx)
  const envKeyModal = createEnvKeyModal(ctx)
  const settingsModal = createSettingsModal(ctx, {
    onOpenKeyModal: () => keyModal.open(),
    onOpenEnvKeyModal: () => void envKeyModal.open(),
    onBackendSwitch: async () => {
      // Refresh data from new backend
      try {
        await refreshIdentity(ctx)
        await refreshJobs(ctx)
        if (ctx.state.snapshot.jobs.length > 0) {
          await selectJob(ctx, ctx.state.snapshot.jobs[0].job_id)
        }
        ctx.state.snapshot.status = `Connected to ${appState.currentBackend}`
      } catch (err: any) {
        ctx.state.snapshot.status = `Switch failed: ${err?.message || "unknown error"}`
      }
      render()
    },
  })
  const snapshotModal = createSnapshotModal(ctx)
  const profileModal = createProfileModal(ctx)
  const urlsModal = createUrlsModal(renderer)
  const usageModal = createUsageModal(ctx)
  const createJobModal = createCreateJobModal(ctx)
  const taskAppsModal = createTaskAppsModal(ctx)
  const logFileModal = createLogFileModal(ctx)
  const sessionsModal = createSessionsModal(ctx)

  const modals = {
    login: loginModal,
    event: eventModal,
    results: resultsModal,
    config: configModal,
    filter: filterModal,
    jobFilter: jobFilterModal,
    key: keyModal,
    settings: settingsModal,
    snapshot: snapshotModal,
    profile: profileModal,
    urls: urlsModal,
    usage: usageModal,
    createJob: createJobModal,
    taskApps: taskAppsModal,
    logFile: logFileModal,
    sessions: sessionsModal,
  }

  // Create keyboard handler
  const handleKeypress = createKeyboardHandler(ctx, modals)

  // Wire up event listeners
  ;(renderer.keyInput as any).on("keypress", handleKeypress)

  // Jobs select widget
  ui.jobsSelect.on(SelectRenderableEvents.SELECTION_CHANGED, (_idx: number, option: any) => {
    if (!option?.value) return
    if (ctx.state.snapshot.selectedJob?.job_id !== option.value) {
      void selectJob(ctx, option.value).then(() => {
        render()
        // Start job details SSE stream for any job type
        maybeStartJobDetailsStream(option.value)
      }).catch(() => {})
    }
  })

  // Helper to start job details stream for any job
  function maybeStartJobDetailsStream(jobId: string): void {
    try {
      // Disconnect existing stream first
      disconnectJobDetailsSse()

      const job = ctx.state.snapshot.selectedJob
      if (job) {
        startJobDetailsStream(jobId)
      }
    } catch {
      // Ignore errors - don't crash
    }
  }

  // Set up focus manager with jobsSelect as default
  focusManager.setDefault({
    id: "jobs-pane",
    onFocus: () => ui.jobsSelect.focus(),
    onBlur: () => ui.jobsSelect.blur(),
  })

  // Initialize pane focusables for events/logs navigation
  initPaneFocusables(ctx, modals.event.open, modals.logFile.open)

  // Start renderer
  renderer.start()
  render()

  // Keep logs pane in sync with filesystem updates
  registerInterval(
    setInterval(() => {
      if (ctx.state.appState.activePane === "logs") {
        render()
      }
    }, 1000)
  )

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
    void refreshHealth(ctx).catch(() => {})
    await refreshIdentity(ctx)
    await refreshJobs(ctx)

    // Load tunnels (task apps) and check their health
    await refreshTunnels(ctx)

    // Auto-start OpenCode server in background
    ctx.state.snapshot.status = "Starting OpenCode server..."
    render()
    const openCodeUrl = await startOpenCodeServer()
    if (openCodeUrl) {
      ctx.state.snapshot.status = `OpenCode ready at ${openCodeUrl}`
      // Auto-create a session for the local server
      appState.openCodeSessionId = `local-${Date.now()}`
    } else {
      ctx.state.snapshot.status = "OpenCode not available (install with: npm i -g opencode)"
    }
    void refreshTunnelHealth(ctx).then(() => render()).catch(() => {})

    const { initialJobId } = ctx.state.config
    if (initialJobId) {
      await selectJob(ctx, initialJobId)
      maybeStartJobDetailsStream(initialJobId)
    } else if (ctx.state.snapshot.jobs.length > 0) {
      const firstJobId = ctx.state.snapshot.jobs[0].job_id
      await selectJob(ctx, firstJobId)
      maybeStartJobDetailsStream(firstJobId)
    }

    // Start SSE connection for real-time job updates
    startJobsStream()

    // Events polling still needed (SSE is only for job list metadata)
    scheduleEventsPoll()
    registerInterval(setInterval(() => void refreshHealth(ctx).catch(() => {}), 30_000))
    registerInterval(setInterval(() => void refreshIdentity(ctx).then(() => render()).catch(() => {}), 60_000))
    // Refresh tunnels every 30 seconds, health checks every 15 seconds
    registerInterval(setInterval(() => void refreshTunnels(ctx).then(() => render()).catch(() => {}), 30_000))
    registerInterval(setInterval(() => void refreshTunnelHealth(ctx).then(() => render()).catch(() => {}), 15_000))

    // Register SSE disconnect for cleanup on shutdown
    registerCleanup("sse", () => pollingState.sseDisconnect?.())
    registerCleanup("job-details-sse", () => disconnectJobDetailsSse())

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
        best_reward: null,
        best_snapshot_id: null,
        total_tokens: null,
        total_cost_usd: null,
        error: event.error ?? null,
        job_source: event.job_type === "prompt_learning" ? JobSource.PromptLearning : JobSource.Learning,
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
      unregisterTimeout(pollingState.sseReconnectTimer)
    }
    pollingState.sseReconnectTimer = registerTimeout(setTimeout(() => {
      pollingState.sseReconnectTimer = null
      startJobsStream()
    }, pollingState.sseReconnectDelay))

    // Exponential backoff: 1s, 2s, 4s, ... up to 30s
    pollingState.sseReconnectDelay = Math.min(pollingState.sseReconnectDelay * 2, 30_000)
  }

  // SSE stream for job details (works for all job types)
  let jobDetailsStreamConnection: JobDetailsStreamConnection | null = null

  function startJobDetailsStream(jobId: string): void {
    try {
      const { pollingState } = ctx.state

      if (!process.env.SYNTH_API_KEY) {
        // No API key, fall back to polling
        return
      }

      // Disconnect any existing stream
      if (jobDetailsStreamConnection) {
        try {
          jobDetailsStreamConnection.disconnect()
        } catch {
          // Ignore disconnect errors
        }
        jobDetailsStreamConnection = null
      }

      jobDetailsStreamConnection = connectJobDetailsStream(
        jobId,
        (event) => handleJobDetailsStreamEvent(event),
        (err) => handleJobDetailsStreamError(err, jobId),
        pollingState.lastJobDetailsSseSeq,
      )

      // SSE connected - mark as connected
      pollingState.jobDetailsSseConnected = true
      pollingState.jobDetailsSseJobId = jobId
      pollingState.jobDetailsSseDisconnect = jobDetailsStreamConnection.disconnect
      pollingState.jobDetailsSseReconnectDelay = 1000 // Reset backoff
    } catch {
      // Failed to start stream - fall back to polling (already scheduled)
    }
  }

  function handleJobDetailsStreamEvent(event: JobDetailsStreamEvent): void {
    try {
      const { snapshot, pollingState, config: appConfig } = ctx.state

      // Ignore if job changed
      if (snapshot.selectedJob?.job_id !== event.job_id) return

      // Track sequence for reconnection (handle undefined seq)
      if (typeof event.seq === "number") {
        pollingState.lastJobDetailsSseSeq = event.seq
      }

      // Safely compute timestamp (ts may be null or undefined from connected event)
      const ts = typeof event.ts === "number" && event.ts > 0
        ? new Date(event.ts * 1000).toISOString()
        : new Date().toISOString()

      // Add event to events list
      snapshot.events.push({
        seq: event.seq ?? 0,
        type: event.type ?? "unknown",
        message: event.message ?? "",
        data: event.data ?? {},
        timestamp: ts,
      })

      // Route events by type prefix to job-specific handlers
      const eventType = event.type ?? ""
      if (eventType.startsWith("eval.")) {
        handleEvalEvent(event)
      } else if (eventType.startsWith("learning.")) {
        handleLearningEvent(event)
      } else if (eventType.startsWith("prompt.learning.") || eventType.startsWith("gepa.")) {
        handlePromptLearningEvent(event)
      }

      // Enforce history limit
      const eventHistoryLimit = appConfig.eventHistoryLimit
      if (eventHistoryLimit > 0 && snapshot.events.length > eventHistoryLimit) {
        snapshot.events = snapshot.events.slice(-eventHistoryLimit)
      }

      render()
    } catch {
      // Swallow errors in event handler - don't crash the app
    }
  }

  // Handle eval job events
  function handleEvalEvent(event: JobDetailsStreamEvent): void {
    const { snapshot } = ctx.state
    const data = event.data ?? {}

    switch (event.type) {
      case "eval.results.updated":
      case "eval.seed.completed": {
        // Update or add result row
        const seed = data.seed as number | undefined
        if (typeof seed === "number") {
          const existingIdx = snapshot.jobDetails.resultRows.findIndex(
            (r) => r.id === seed
          )
          const resultRow: JobResultRow = {
            id: seed,
            score: (data.score ?? data.outcome_reward ?? null) as number | null,
            reward: (data.outcome_reward ?? null) as number | null,
            latency_ms: (data.latency_ms ?? null) as number | null,
            tokens: (data.tokens ?? null) as number | null,
            cost_usd: (data.cost_usd ?? null) as number | null,
            error: (data.error ?? null) as string | null,
            trace_id: (data.trace_id ?? null) as string | null,
            metadata: {
              trial_id: data.trial_id,
              correlation_id: data.correlation_id,
              events_score: data.events_score,
              verifier_score: data.verifier_score,
            },
          }
          if (existingIdx !== -1) {
            snapshot.jobDetails.resultRows[existingIdx] = resultRow
          } else {
            snapshot.jobDetails.resultRows.push(resultRow)
            snapshot.jobDetails.resultRows.sort((a, b) =>
              (typeof a.id === "number" ? a.id : 0) - (typeof b.id === "number" ? b.id : 0)
            )
          }
        }
        break
      }

      case "eval.job.progress": {
        // Update status with progress
        const completed = (data.completed ?? 0) as number
        const total = (data.total ?? 0) as number
        snapshot.status = `Eval: ${completed}/${total} seeds completed`
        break
      }

      case "eval.job.completed": {
        // Update summary and job status
        snapshot.jobDetails.summary = {
          completed: data.completed as number | undefined,
          total: data.total as number | undefined,
          failed: data.failed as number | undefined,
          mean_reward: (data.mean_reward ?? null) as number | null,
          ...data,
        }
        if (snapshot.selectedJob) {
          snapshot.selectedJob.status = data.error ? "failed" : "completed"
          if (data.error) {
            snapshot.selectedJob.error = String(data.error)
          }
        }
        const meanReward = data.mean_reward as number | undefined
        snapshot.status = data.error
          ? `Eval failed: ${String(data.error).slice(0, 50)}`
          : `Eval completed: ${data.completed ?? 0}/${data.total ?? 0} seeds (mean reward: ${meanReward?.toFixed(3) ?? "N/A"})`
        break
      }

      case "eval.job.started": {
        snapshot.status = `Eval started: ${data.seed_count ?? 0} seeds`
        break
      }
    }
  }

  // Handle learning job events
  function handleLearningEvent(event: JobDetailsStreamEvent): void {
    const { snapshot } = ctx.state
    const data = event.data ?? {}

    // Handle learning.iteration.completed and similar events
    if (event.type === "learning.iteration.completed") {
      const resultRow: JobResultRow = {
        id: data.iteration as string | number | undefined,
        score: (data.reward ?? data.score ?? null) as number | null,
        reward: (data.reward ?? null) as number | null,
        latency_ms: (data.latency_ms ?? null) as number | null,
        tokens: (data.tokens ?? null) as number | null,
        cost_usd: (data.cost_usd ?? null) as number | null,
        error: (data.error ?? null) as string | null,
        metadata: data,
      }
      snapshot.jobDetails.resultRows.push(resultRow)
    }

    // Handle progress events
    if (event.type === "learning.job.progress") {
      if (snapshot.jobDetails.summary) {
        snapshot.jobDetails.summary.completed = data.completed as number | undefined
        snapshot.jobDetails.summary.total = data.total as number | undefined
      } else {
        snapshot.jobDetails.summary = {
          completed: data.completed as number | undefined,
          total: data.total as number | undefined,
        }
      }
    }
  }

  // Handle prompt-learning job events
  function handlePromptLearningEvent(event: JobDetailsStreamEvent): void {
    const { snapshot } = ctx.state
    const data = event.data ?? {}

    // Handle progress events
    if (event.type === "prompt.learning.progress" || event.type === "gepa.progress") {
      if (snapshot.jobDetails.summary) {
        snapshot.jobDetails.summary.completed = data.completed as number | undefined
        snapshot.jobDetails.summary.total = data.total as number | undefined
      } else {
        snapshot.jobDetails.summary = {
          completed: data.completed as number | undefined,
          total: data.total as number | undefined,
        }
      }
    }

    // Handle trial results
    if (event.type === "prompt.learning.trial.results") {
      const resultRow: JobResultRow = {
        id: data.trial_id as string | undefined,
        score: (data.reward ?? data.score ?? null) as number | null,
        reward: (data.reward ?? null) as number | null,
        tokens: (data.tokens ?? null) as number | null,
        cost_usd: (data.cost_usd ?? null) as number | null,
        error: (data.error ?? null) as string | null,
        metadata: data,
      }
      snapshot.jobDetails.resultRows.push(resultRow)
    }
  }

  function handleJobDetailsStreamError(_err: Error, jobId: string): void {
    try {
      const { pollingState } = ctx.state

      pollingState.jobDetailsSseConnected = false
      pollingState.jobDetailsSseDisconnect = null
      jobDetailsStreamConnection = null

      // Resume events polling as fallback
      scheduleEventsPoll()

      // Schedule reconnect with exponential backoff
      clearJobDetailsSseTimer()

      // Reconnect for any job type (not just eval)
      if (ctx.state.snapshot.selectedJob?.job_id === jobId) {
        pollingState.jobDetailsSseReconnectTimer = registerTimeout(setTimeout(() => {
          pollingState.jobDetailsSseReconnectTimer = null
          if (ctx.state.snapshot.selectedJob?.job_id === jobId) {
            startJobDetailsStream(jobId)
          }
        }, pollingState.jobDetailsSseReconnectDelay))

        // Exponential backoff: 1s, 2s, 4s, ... up to 30s
        pollingState.jobDetailsSseReconnectDelay = Math.min(pollingState.jobDetailsSseReconnectDelay * 2, 30_000)
      }
    } catch {
      // Swallow errors in error handler - don't crash the app
    }
  }

  // Polling
  function scheduleJobsPoll(): void {
    const { pollingState } = ctx.state
    // Don't schedule polling if SSE is connected
    if (pollingState.sseConnected) return
    if (pollingState.jobsTimer) {
      clearTimeout(pollingState.jobsTimer)
      unregisterTimeout(pollingState.jobsTimer)
    }
    pollingState.jobsTimer = registerTimeout(setTimeout(pollJobs, pollingState.jobsPollMs))
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
    if (pollingState.eventsTimer) {
      clearTimeout(pollingState.eventsTimer)
      unregisterTimeout(pollingState.eventsTimer)
    }
    pollingState.eventsTimer = registerTimeout(setTimeout(pollEvents, pollingState.eventsPollMs))
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
