import { type Accessor, createEffect, createSignal, onCleanup } from "solid-js"
import fs from "node:fs"
import { scanMultipleDirectories, type LocalApiScanCache, type ScannedLocalAPI } from "./localapi-scanner"

type LocalApiWatchOptions = {
  enabled: Accessor<boolean>
  directories: Accessor<string[]>
  pollIntervalMs?: number
}

export function useLocalApiScanner(options: LocalApiWatchOptions) {
  const [files, setFiles] = createSignal<ScannedLocalAPI[]>([])
  const cache: LocalApiScanCache = new Map()

  createEffect(() => {
    if (!options.enabled()) return

    const dirsToScan = options.directories()
    if (dirsToScan.length === 0) {
      setFiles([])
      return
    }

    let disposed = false
    let refreshTimer: NodeJS.Timeout | null = null
    let pollTimer: NodeJS.Timeout | null = null
    let refreshInFlight = false
    let refreshQueued = false
    const watchers: fs.FSWatcher[] = []

    const scheduleRefresh = () => {
      if (disposed || refreshTimer) return
      refreshTimer = setTimeout(() => {
        refreshTimer = null
        void refreshLocalApis()
      }, 150)
    }

    const refreshLocalApis = async () => {
      if (refreshInFlight) {
        refreshQueued = true
        return
      }
      refreshInFlight = true
      const found = await scanMultipleDirectories(dirsToScan, cache)
      if (!disposed) {
        setFiles(found)
      }
      refreshInFlight = false
      if (refreshQueued) {
        refreshQueued = false
        void refreshLocalApis()
      }
    }

    void refreshLocalApis()
    pollTimer = setInterval(() => {
      void refreshLocalApis()
    }, options.pollIntervalMs ?? 3000)

    for (const dir of dirsToScan) {
      try {
        watchers.push(fs.watch(dir, { persistent: false }, scheduleRefresh))
      } catch {
        // ignore
      }
    }

    onCleanup(() => {
      disposed = true
      if (refreshTimer) clearTimeout(refreshTimer)
      if (pollTimer) clearInterval(pollTimer)
      for (const watcher of watchers) {
        watcher.close()
      }
    })
  })

  return {
    files,
  }
}
