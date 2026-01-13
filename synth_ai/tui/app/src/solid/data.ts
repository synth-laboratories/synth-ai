import { createSignal, onCleanup, onMount, type Accessor } from "solid-js"

import { refreshHealth, refreshIdentity } from "../api/identity"
import { refreshJobs, selectJob } from "../api/jobs"
import { loadPersistedSettings } from "../persistence/settings"
import {
  appState,
  modeKeys,
  normalizeMode,
  switchMode,
} from "../state/app-state"
import { config } from "../state/polling"
import { snapshot } from "../state/snapshot"
import { isLoggedOutMarkerSet, loadSavedApiKey } from "../utils/logout-marker"
import { isOpenCodeServerRunning, startOpenCodeServer } from "../utils/opencode-server"
import { registerCleanup, unregisterCleanup } from "../lifecycle"
import { createSolidContext } from "./context"

export type SolidData = {
  version: Accessor<number>
  refresh: (options?: { signal?: AbortSignal }) => Promise<void>
  select: (jobId: string, options?: { signal?: AbortSignal }) => Promise<void>
  ensureOpenCodeServer: (options?: { signal?: AbortSignal }) => Promise<void>
  ctx: ReturnType<typeof createSolidContext>
}

export function useSolidData(): SolidData {
  const [version, setVersion] = createSignal(0)
  const bump = () => setVersion((current) => current + 1)
  const ctx = createSolidContext(bump)

  async function bootstrap(): Promise<void> {
    // Load persisted mode and keys
    await loadPersistedSettings({
      settingsFilePath: config.settingsFilePath,
      normalizeMode,
      setCurrentMode: (mode) => { appState.currentMode = mode },
      setModeKey: (mode, key) => { modeKeys[mode] = key },
    })

    // Apply current mode's URLs to env
    switchMode(appState.currentMode)
    bump()

    if (isLoggedOutMarkerSet()) {
      snapshot.status = "Sign in required"
      bump()
      return
    }

    // Try loading saved key if not set
    if (!process.env.SYNTH_API_KEY) {
      const savedKey = loadSavedApiKey()
      if (savedKey) {
        process.env.SYNTH_API_KEY = savedKey
        modeKeys[appState.currentMode] = savedKey
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

  async function refresh(options: { signal?: AbortSignal } = {}): Promise<void> {
    if (isLoggedOutMarkerSet() || !process.env.SYNTH_API_KEY) {
      return
    }
    await refreshJobs(ctx, options)
    await refreshHealth(ctx, options)
    if (options.signal?.aborted) return
    bump()
  }

  async function select(jobId: string, options: { signal?: AbortSignal } = {}): Promise<void> {
    await selectJob(ctx, jobId, options)
    if (options.signal?.aborted) return
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

  async function ensureOpenCodeServer(options: { signal?: AbortSignal } = {}): Promise<void> {
    if (options.signal?.aborted) {
      return
    }
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
    if (options.signal?.aborted) {
      return
    }
    if (openCodeUrl) {
      setOpenCodeStatus(`ready at ${openCodeUrl}`)
      bump()
      return
    }

    if (isOpenCodeServerRunning()) {
      const delayedUrl = await waitForOpenCodeUrl(60000)
      if (options.signal?.aborted) {
        return
      }
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
    const interval = setInterval(() => {
      void refresh()
    }, Math.max(1, config.refreshInterval) * 1000)
    const cleanupName = "data-refresh-interval"
    const cleanup = () => {
      clearInterval(interval)
    }
    registerCleanup(cleanupName, cleanup)
    onCleanup(() => {
      cleanup()
      unregisterCleanup(cleanupName)
    })
  })

  return {
    version,
    refresh,
    select,
    ensureOpenCodeServer,
    ctx,
  }
}
