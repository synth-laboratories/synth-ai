import { createEffect, onCleanup, onMount } from "solid-js"

import { refreshHealth, refreshIdentity } from "../api/identity"
import { refreshJobs, selectJob } from "../api/jobs"
import { readPersistedSettings } from "../persistence/settings"
import { getJobsCacheKey, loadJobsCache } from "../persistence/jobs-cache"
import { switchMode, setCurrentMode } from "../state/app-state"
import { config, pollingState, setPollNextAt, shouldPoll, clearJobsTimer, onSseChange } from "../state/polling"
import { isOpenCodeServerRunning, startOpenCodeServer } from "../utils/opencode-server"
import { registerCleanup, unregisterCleanup } from "../lifecycle"
import { isAborted } from "../utils/request"
import { createSolidContext } from "./context"
import { createAppStore } from "./store"
import { log } from "../utils/log"
import { ListPane, type Mode } from "../types"

export type SolidData = {
  refresh: () => Promise<boolean>
  select: (jobId: string) => Promise<void>
  ensureOpenCodeServer: () => Promise<void>
  ctx: ReturnType<typeof createSolidContext>
}

export function useSolidData(): SolidData {
  const { data, setData, ui, setUi } = createAppStore()
  const ctx = createSolidContext(data, setData, ui, setUi)

  createEffect(() => {
    const orgId = data.orgId
    if (!orgId) {
      if (data.jobsCacheKey) {
        setData("jobsCacheKey", null)
        setData("jobsCache", [])
        setData("jobsCacheAppended", [])
      }
      if (data.jobsLoaded) {
        setData("jobsLoaded", false)
      }
      return
    }

    const cacheKey = getJobsCacheKey(orgId, ui.currentMode)
    if (data.jobsCacheKey === cacheKey) return

    void loadJobsCache(cacheKey).then((jobs) => {
      if (isAborted()) return
      setData("jobsCache", jobs)
      setData("jobsCacheKey", cacheKey)
      setData("jobsCacheAppended", [])
      log("state", "jobs cache loaded", { key: cacheKey, count: jobs.length })
    })
  })

  async function bootstrap(): Promise<void> {
    const settings = await readPersistedSettings()
    const persistedMode = settings.mode ?? null
    const persistedPrimaryView = settings.primaryView ?? null
    let resolvedMode: Mode = ui.currentMode
    // Persisted settings are the source of truth for mode.
    if (persistedMode) {
      resolvedMode = persistedMode
    }
    setUi("settingsMode", persistedMode)
    setCurrentMode(resolvedMode)
    setUi("currentMode", resolvedMode)
    if (persistedMode) {
      switchMode(resolvedMode)
    }
    if (persistedPrimaryView) {
      setUi("primaryView", persistedPrimaryView)
    }
    const listFilters = settings.listFilters?.[resolvedMode]
    if (listFilters) {
      setUi("listFilterMode", ListPane.Jobs, listFilters[ListPane.Jobs].mode)
      setUi("listFilterSelections", ListPane.Jobs, new Set(listFilters[ListPane.Jobs].selections))
      setUi("listFilterMode", ListPane.Logs, listFilters[ListPane.Logs].mode)
      setUi("listFilterSelections", ListPane.Logs, new Set(listFilters[ListPane.Logs].selections))
      setUi("listFilterMode", ListPane.Sessions, listFilters[ListPane.Sessions].mode)
      setUi("listFilterSelections", ListPane.Sessions, new Set(listFilters[ListPane.Sessions].selections))
    }
    setUi("settingsLoaded", true)
    if (!process.env.SYNTH_API_KEY) {
      process.env.SYNTH_API_KEY = settings.keys[resolvedMode] || ""
    }

    if (!process.env.SYNTH_API_KEY) {
      setData("status", "Sign in required")
      return
    }

    // Check identity first - this validates the API key
    const identityResult = await refreshIdentity(ctx)
    if (identityResult.authError) {
      // API key is invalid, signal re-auth needed
      setData("status", "Sign in required")
      return
    }

    const jobsPromise = refreshJobs(ctx, { limit: config.jobLimit })
    const healthPromise = refreshHealth(ctx, { force: true })
    const results = await Promise.allSettled([healthPromise, jobsPromise])
    const jobsResult = results[1]
    if (jobsResult.status === "fulfilled") {
      setData("jobsLoaded", true)
    }
  }

  async function refresh(): Promise<boolean> {
    if (!process.env.SYNTH_API_KEY) {
      return false
    }
    const [jobsResult] = await Promise.all([
      refreshJobs(ctx),
      refreshHealth(ctx, { force: true }),
    ])
    if (isAborted()) return false
    if (jobsResult.ok) {
      setData("jobsLoaded", true)
    }
    return jobsResult.ok
  }

  async function refreshPolling(): Promise<boolean> {
    if (!process.env.SYNTH_API_KEY) {
      return false
    }
    let jobsOk = true
    if (shouldPoll("jobs")) {
      const jobsResult = await refreshJobs(ctx)
      jobsOk = jobsResult.ok
      if (jobsOk) {
        setData("jobsLoaded", true)
      }
      if (!jobsOk) {
        void refreshHealth(ctx)
      }
    }
    if (isAborted()) return false
    return jobsOk
  }

  async function select(jobId: string): Promise<void> {
    await selectJob(ctx, jobId)
    if (isAborted()) return
  }

  function setOpenCodeStatus(message: string): void {
    setUi("openCodeStatus", message)
  }

  async function waitForOpenCodeUrl(timeoutMs: number): Promise<string | null> {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (ui.openCodeUrl) {
        return ui.openCodeUrl
      }
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
    return null
  }

  async function ensureOpenCodeServer(): Promise<void> {
    if (isAborted()) {
      return
    }
    if (ui.openCodeUrl) {
      setOpenCodeStatus(`ready at ${ui.openCodeUrl}`)
      return
    }
    if (process.env.OPENCODE_URL) {
      setOpenCodeStatus(`ready at ${process.env.OPENCODE_URL}`)
      return
    }

    setOpenCodeStatus("starting... (first run may take a minute)")
    const openCodeUrl = await startOpenCodeServer({
      onUrl: (url) => {
        setUi("openCodeUrl", url)
      },
    })
    if (isAborted()) {
      return
    }
    if (openCodeUrl) {
      setOpenCodeStatus(`ready at ${openCodeUrl}`)
      return
    }

    if (isOpenCodeServerRunning()) {
      const delayedUrl = await waitForOpenCodeUrl(60000)
      if (isAborted()) {
        return
      }
      if (delayedUrl) {
        setOpenCodeStatus(`ready at ${delayedUrl}`)
        return
      }
    }

    setOpenCodeStatus(
      "not available (install: brew install synth-laboratories/tap/opencode-synth, npm i -g opencode, or set OPENCODE_DEV_PATH)",
    )
  }

  onMount(() => {
    void bootstrap()
    let delayMs = Math.max(1, config.refreshInterval) * 1000
    const minDelayMs = delayMs
    const maxDelayMs = Math.max(delayMs, Math.max(1, config.maxRefreshInterval) * 1000)

    const schedule = (nextDelay: number) => {
      if (!shouldPoll("jobs")) {
        clearJobsTimer()
        return
      }
      if (pollingState.jobsTimer) clearTimeout(pollingState.jobsTimer)
      const delay = Math.max(0, nextDelay)
      setPollNextAt("jobs", Date.now() + delay)
      pollingState.jobsTimer = setTimeout(() => {
        void run()
      }, delay)
    }

    const run = async () => {
      if (!shouldPoll("jobs")) {
        clearJobsTimer()
        return
      }
      if (pollingState.jobsInFlight) {
        schedule(delayMs)
        return
      }
      pollingState.jobsInFlight = true
      let ok = false
      try {
        ok = await refreshPolling().catch(() => false)
      } finally {
        pollingState.jobsInFlight = false
      }
      if (!shouldPoll("jobs")) {
        clearJobsTimer()
        return
      }
      if (ok) {
        delayMs = minDelayMs
      } else {
        delayMs = Math.min(maxDelayMs, Math.floor(delayMs * 1.7))
      }
      schedule(delayMs)
    }

    const unsubscribeSse = onSseChange("jobs", (connected) => {
      if (connected) {
        clearJobsTimer()
        return
      }
      delayMs = minDelayMs
      schedule(0)
    })
    if (shouldPoll("jobs")) {
      schedule(delayMs)
    }
    const cleanupName = "data-refresh-interval"
    const cleanup = () => {
      clearJobsTimer()
      pollingState.jobsInFlight = false
      unsubscribeSse()
    }
    registerCleanup(cleanupName, cleanup)
    onCleanup(() => {
      cleanup()
      unregisterCleanup(cleanupName)
    })
  })

  return {
    refresh,
    select,
    ensureOpenCodeServer,
    ctx,
  }
}
