import { type Accessor, createEffect, createSignal, onCleanup } from "solid-js"
import fs from "node:fs"
import { getLogsDirectory, listLogFiles, readLogChunk, type LogFileInfo } from "./logs"

type LiveLogsOptions = {
  listActive: Accessor<boolean>
  detailActive: Accessor<boolean>
  onSelectionAdjusted?: () => void
}

type LiveLogsState = {
  files: Accessor<LogFileInfo[]>
  lines: Accessor<string[]>
  selectedIndex: Accessor<number>
  setSelectedIndex: (next: number) => void
  moveSelection: (delta: number) => boolean
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function filesEqual(a: LogFileInfo[], b: LogFileInfo[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i].path !== b[i].path || a[i].mtimeMs !== b[i].mtimeMs || a[i].size !== b[i].size) {
      return false
    }
  }
  return true
}

export function useLiveLogs(options: LiveLogsOptions): LiveLogsState {
  const [files, setFiles] = createSignal<LogFileInfo[]>([])
  const [lines, setLines] = createSignal<string[]>([])
  const [selectedIndex, setSelectedIndexState] = createSignal(0)

  const clampSelection = (next: number, list: LogFileInfo[]) => {
    if (!list.length) return 0
    return clamp(next, 0, list.length - 1)
  }

  const setSelectedIndex = (next: number) => {
    const list = files()
    const clamped = clampSelection(next, list)
    if (clamped !== selectedIndex()) {
      setSelectedIndexState(clamped)
    }
  }

  const moveSelection = (delta: number): boolean => {
    const list = files()
    if (!list.length) return false
    const next = clampSelection(selectedIndex() + delta, list)
    if (next === selectedIndex()) return false
    setSelectedIndexState(next)
    return true
  }

  createEffect(() => {
    if (!options.listActive()) return

    let disposed = false
    let refreshTimer: NodeJS.Timeout | null = null
    let pollTimer: NodeJS.Timeout | null = null
    let watcher: fs.FSWatcher | null = null
    let refreshInFlight = false
    let refreshQueued = false

    const scheduleRefresh = () => {
      if (disposed || refreshTimer) return
      refreshTimer = setTimeout(() => {
        refreshTimer = null
        void refreshLogFiles()
      }, 100)
    }

    const refreshLogFiles = async () => {
      if (refreshInFlight) {
        refreshQueued = true
        return
      }
      refreshInFlight = true

      const previous = files()
      const selectedPath = previous[selectedIndex()]?.path ?? null
      const nextFiles = await listLogFiles()
      if (disposed) return

      if (!filesEqual(previous, nextFiles)) {
        setFiles(nextFiles)
      }

      let nextIndex = selectedIndex()
      if (!nextFiles.length) {
        nextIndex = 0
      } else if (selectedPath) {
        const idx = nextFiles.findIndex((file) => file.path === selectedPath)
        nextIndex = idx >= 0 ? idx : clampSelection(nextIndex, nextFiles)
      } else {
        nextIndex = clampSelection(nextIndex, nextFiles)
      }

      if (nextIndex !== selectedIndex()) {
        setSelectedIndexState(nextIndex)
        options.onSelectionAdjusted?.()
      }

      refreshInFlight = false
      if (refreshQueued) {
        refreshQueued = false
        void refreshLogFiles()
      }
    }

    void refreshLogFiles()
    pollTimer = setInterval(() => {
      void refreshLogFiles()
    }, 2000)

    try {
      watcher = fs.watch(getLogsDirectory(), { persistent: false }, scheduleRefresh)
    } catch {
      // ignore
    }

    onCleanup(() => {
      disposed = true
      if (refreshTimer) clearTimeout(refreshTimer)
      if (pollTimer) clearInterval(pollTimer)
      watcher?.close()
    })
  })

  createEffect(() => {
    if (!options.detailActive()) {
      setLines([])
      return
    }

    const list = files()
    const selected = selectedIndex()
    const file = list[selected]
    if (!file) {
      setLines([])
      return
    }

    let disposed = false
    let refreshTimer: NodeJS.Timeout | null = null
    let pollTimer: NodeJS.Timeout | null = null
    let watcher: fs.FSWatcher | null = null
    let refreshInFlight = false
    let refreshQueued = false
    let tailState = { size: 0, remainder: "" }

    const appendText = (text: string, reset: boolean) => {
      if (disposed) return
      setLines((current) => {
        const base = reset ? [] : current
        const combined = `${reset ? "" : tailState.remainder}${text}`
        const parts = combined.split("\n")
        tailState.remainder = combined.endsWith("\n") ? "" : (parts.pop() ?? "")
        return reset ? parts : [...base, ...parts]
      })
    }

    const loadFull = async () => {
      const chunk = await readLogChunk(file.path, 0)
      if (!chunk || disposed) return
      tailState.size = chunk.size
      tailState.remainder = ""
      appendText(chunk.text, true)
    }

    const loadDelta = async () => {
      const chunk = await readLogChunk(file.path, tailState.size)
      if (!chunk || disposed) return
      if (chunk.truncated) {
        await loadFull()
        return
      }
      tailState.size = chunk.size
      if (chunk.text) {
        appendText(chunk.text, false)
      }
    }

    const refreshContent = async (force: boolean) => {
      if (refreshInFlight) {
        refreshQueued = true
        return
      }
      refreshInFlight = true
      if (force) {
        await loadFull()
      } else {
        await loadDelta()
      }
      refreshInFlight = false
      if (refreshQueued) {
        refreshQueued = false
        void refreshContent(false)
      }
    }

    const scheduleRefresh = () => {
      if (disposed || refreshTimer) return
      refreshTimer = setTimeout(() => {
        refreshTimer = null
        void refreshContent(false)
      }, 100)
    }

    void refreshContent(true)
    pollTimer = setInterval(() => {
      void refreshContent(false)
    }, 1000)
    try {
      watcher = fs.watch(file.path, { persistent: false }, scheduleRefresh)
    } catch {
      // ignore
    }

    onCleanup(() => {
      disposed = true
      if (refreshTimer) clearTimeout(refreshTimer)
      if (pollTimer) clearInterval(pollTimer)
      watcher?.close()
    })
  })

  return {
    files,
    lines,
    selectedIndex,
    setSelectedIndex,
    moveSelection,
  }
}
