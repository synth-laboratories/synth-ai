import { createEffect, onCleanup, onMount } from "solid-js"

import { refreshHealth, refreshIdentity } from "../api/identity"
import { refreshJobs, selectJob } from "../api/jobs"
import { loadPersistedSettings } from "../persistence/settings"
import { getJobsCacheKey, loadJobsCache } from "../persistence/jobs-cache"
import { switchMode, setCurrentMode } from "../state/app-state"
import { config } from "../state/polling"
import { isOpenCodeServerRunning, startOpenCodeServer } from "../utils/opencode-server"
import { registerCleanup, unregisterCleanup } from "../lifecycle"
import { isAborted } from "../utils/request"
import { createSolidContext } from "./context"
import { createAppStore } from "./store"
import { log } from "../utils/log"

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
    let persisted: Awaited<ReturnType<typeof loadPersistedSettings>> | null = null
    // If mode is explicitly set via env, use it directly.
    // Otherwise fall back to persisted settings.
    if (!process.env.SYNTH_TUI_MODE) {
      persisted = await loadPersistedSettings({
        setCurrentMode: (mode) => {
          setCurrentMode(mode)
          setUi("currentMode", mode)
        },
      })
      // Apply loaded mode's URLs
      switchMode(ui.currentMode)
    }
    if (!process.env.SYNTH_API_KEY) {
      const settings =
        persisted ??
        (await loadPersistedSettings({
          setCurrentMode: () => {},
        }))
      process.env.SYNTH_API_KEY = settings.keys[ui.currentMode] || ""
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

    const tasks = [
      refreshHealth(ctx),
      refreshJobs(ctx),
    ]
    await Promise.allSettled(tasks)
  }

  async function refresh(): Promise<boolean> {
    if (!process.env.SYNTH_API_KEY) {
      return false
    }
    const jobsResult = await refreshJobs(ctx)
    await refreshHealth(ctx)
    if (isAborted()) return false
    return jobsResult.ok
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
    let timer: ReturnType<typeof setTimeout> | null = null
    let inFlight = false

    const schedule = (nextDelay: number) => {
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => {
        void run()
      }, nextDelay)
    }

    const run = async () => {
      if (inFlight) {
        schedule(delayMs)
        return
      }
      inFlight = true
      const ok = await refresh().catch(() => false)
      inFlight = false
      if (ok) {
        delayMs = minDelayMs
      } else {
        delayMs = Math.min(maxDelayMs, Math.floor(delayMs * 1.7))
      }
      schedule(delayMs)
    }

    schedule(delayMs)
    const cleanupName = "data-refresh-interval"
    const cleanup = () => {
      if (timer) clearTimeout(timer)
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
