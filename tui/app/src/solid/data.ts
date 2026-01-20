import { createSignal, onCleanup, onMount, type Accessor } from "solid-js"

import { refreshHealth, refreshIdentity } from "../api/identity"
import { refreshJobs, selectJob, fetchApiCandidates } from "../api/jobs"
import { loadPersistedSettings } from "../persistence/settings"
import {
  appState,
  backendConfigs,
  frontendKeys,
  frontendKeySources,
  getKeyForBackend,
  normalizeBackendId,
} from "../state/app-state"
import { config } from "../state/polling"
import { snapshot } from "../state/snapshot"
import { isLoggedOutMarkerSet, loadSavedApiKey } from "../utils/logout-marker"
import { isOpenCodeServerRunning, startOpenCodeServer } from "../utils/opencode-server"
import { createSolidContext } from "./context"

export type SolidData = {
  version: Accessor<number>
  refresh: () => Promise<void>
  select: (jobId: string) => Promise<void>
  ctx: ReturnType<typeof createSolidContext>
  installOpenCode: () => Promise<void>
}

export function useSolidData(): SolidData {
  const [version, setVersion] = createSignal(0)
  const bump = () => setVersion((current) => current + 1)
  const ctx = createSolidContext(bump)

  async function bootstrap(): Promise<void> {
    await loadPersistedSettings({
      settingsFilePath: config.settingsFilePath,
      normalizeBackendId,
      setCurrentBackend: (id) => { appState.currentBackend = id },
      setFrontendKey: (id, key) => { frontendKeys[id] = key },
      setFrontendKeySource: (id, source) => { frontendKeySources[id] = source },
    })

    const currentConfig = backendConfigs[appState.currentBackend]
    // Ensure the JS app uses the backend selected in persisted settings / UI.
    //
    // Previously we only set SYNTH_BACKEND_URL if it was missing, which meant the Python launcher
    // could pin the TUI to a different backend (often prod) even when SYNTH_TUI_BACKEND="local".
    // That leads to confusing symptoms like:
    // - job IDs visible in the TUI that don't exist on the local backend
    // - "missing recent jobs" when you're actually looking at a different org/backend
    //
    // The launcher can still control the chosen backend by setting SYNTH_TUI_BACKEND and/or
    // SYNTH_TUI_*_API_BASE env vars; but we should never silently disagree with appState.currentBackend.
    process.env.SYNTH_BACKEND_URL = currentConfig.baseUrl.replace(/\/api$/, "")
    if (!process.env.SYNTH_API_KEY || !process.env.SYNTH_API_KEY.trim()) {
      process.env.SYNTH_API_KEY = getKeyForBackend(appState.currentBackend) || process.env.SYNTH_API_KEY || ""
    }
    bump()

    if (isLoggedOutMarkerSet()) {
      snapshot.status = "Sign in required"
      bump()
      return
    }

    if (!process.env.SYNTH_API_KEY) {
      const savedKey = loadSavedApiKey()
      if (savedKey) {
        process.env.SYNTH_API_KEY = savedKey
      }
    }

    if (!process.env.SYNTH_API_KEY) {
      snapshot.status = "Sign in required"
      bump()
      return
    }

    await refreshIdentity(ctx)
    await refreshHealth(ctx)
    await refreshJobs(ctx)
    bump()
  }

  async function refresh(): Promise<void> {
    if (isLoggedOutMarkerSet() || !process.env.SYNTH_API_KEY) {
      return
    }
    await refreshJobs(ctx)
    await refreshHealth(ctx)

    // Poll candidates for active jobs to show live optimization progress
    const job = snapshot.selectedJob
    if (job && isJobActive(job)) {
      fetchApiCandidates(ctx, appState.jobSelectToken).catch(() => {
        // Silently ignore - candidates are supplementary
      })
    }

    bump()
  }

  function isJobActive(job: { status?: string | null }): boolean {
    const status = job.status?.toLowerCase()
    return status === "pending" || status === "running" || status === "in_progress"
  }

  async function select(jobId: string): Promise<void> {
    await selectJob(ctx, jobId)
    bump()
  }

  function setOpenCodeStatus(message: string): void {
    appState.openCodeStatus = message
  }

  async function waitForOpenCodeUrl(timeoutMs: number): Promise<string | null> {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (appState.openCodeUrl) {
        return appState.openCodeUrl
      }
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
    return null
  }

  async function ensureOpenCodeServer(): Promise<void> {
    if (appState.openCodeUrl) {
      setOpenCodeStatus(`ready at ${appState.openCodeUrl}`)
      bump()
      return
    }
    if (process.env.OPENCODE_URL) {
      setOpenCodeStatus(`ready at ${process.env.OPENCODE_URL}`)
      bump()
      return
    }

    setOpenCodeStatus("starting... (first run may take a minute)")
    bump()
    const openCodeUrl = await startOpenCodeServer()
    if (openCodeUrl) {
      setOpenCodeStatus(`ready at ${openCodeUrl}`)
      bump()
      return
    }

    if (isOpenCodeServerRunning()) {
      const delayedUrl = await waitForOpenCodeUrl(60000)
      if (delayedUrl) {
        setOpenCodeStatus(`ready at ${delayedUrl}`)
        bump()
        return
      }
    }

    setOpenCodeStatus(
      "not available (install: brew install synth-laboratories/tap/opencode-synth, npm i -g opencode, or set OPENCODE_DEV_PATH)",
    )
    bump()
  }

  onMount(() => {
    void bootstrap()
    void ensureOpenCodeServer()
    const interval = setInterval(() => {
      void refresh()
    }, Math.max(1, config.refreshInterval) * 1000)
    onCleanup(() => clearInterval(interval))
  })

  return {
    version,
    refresh,
    select,
    ctx,
    // For now "install" just re-runs the OpenCode startup flow; actual installation is out of scope.
    installOpenCode: ensureOpenCodeServer,
  }
}
