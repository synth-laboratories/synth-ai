import { createEffect, createSignal, onCleanup, type Accessor } from "solid-js"

import { refreshEvents } from "../../api/events"
import type { JobDetailsStreamEvent } from "../../api/job-details-stream"
import type { ActivePane, PrincipalPane } from "../../types"
import type { AppContext } from "../../context"
import { registerCleanup, unregisterCleanup } from "../../lifecycle"
import { useJobDetailsStream } from "../api/useJobDetailsStream"
import type { JobEvent } from "../../tui_data"

type UseJobEventsOptions = {
  ctx: AppContext
  selectedJobId: Accessor<string | null>
  activePane: Accessor<ActivePane>
  principalPane: Accessor<PrincipalPane>
}

export function useJobEvents(options: UseJobEventsOptions): void {
  const { snapshot, appState } = options.ctx.state
  const [lastSeenSeq, setLastSeenSeq] = createSignal(0)

  // Subscribe to real-time job details updates via SSE.
  useJobDetailsStream({
    jobId: options.selectedJobId,
    sinceSeq: lastSeenSeq,
    enabled: () => options.principalPane() === "jobs" && options.activePane() !== "logs",
    onEvent: (event: JobDetailsStreamEvent) => {
      if (event.seq > lastSeenSeq()) {
        setLastSeenSeq(event.seq)
      }
      // Keep polling cursor in sync with SSE cursor so refreshEvents() doesn't refetch from 0.
      appState.lastSeq = Math.max(appState.lastSeq || 0, event.seq)

      const jobEvent: JobEvent = {
        seq: event.seq,
        type: event.type,
        message: event.message,
        data: event.data as JobEvent["data"],
        timestamp: new Date(event.ts).toISOString(),
      }

      const existingSeqs = new Set(snapshot.events.map((e) => e.seq))
      if (!existingSeqs.has(jobEvent.seq)) {
        snapshot.events = [...snapshot.events, jobEvent].sort((a, b) => a.seq - b.seq)
        options.ctx.render()
      }
    },
    onError: (error) => {
      // Log but don't show to user - polling will still work as fallback.
      console.error("Job details SSE error:", error.message)
    },
  })

  // Poll/backfill events to avoid gaps when SSE drops or when selecting an older job.
  createEffect(() => {
    const jobId = options.selectedJobId()
    const enabled = options.principalPane() === "jobs" && options.activePane() !== "logs"
    if (!jobId || !enabled) return

    let cancelled = false

    async function backfillOnce(): Promise<void> {
      // Pull up to ~10 pages (2000 events) max per selection, but stop early if no progress.
      for (let i = 0; i < 10; i++) {
        const beforeSeq = appState.lastSeq
        const beforeLen = snapshot.events.length
        const ok = await refreshEvents(options.ctx)
        if (!ok) break
        if (cancelled) return
        if (snapshot.events.length === beforeLen && appState.lastSeq === beforeSeq) break
      }
      if (!cancelled) options.ctx.render()
    }

    void backfillOnce()

    const interval = setInterval(() => {
      void refreshEvents(options.ctx).then((ok) => {
        if (ok && !cancelled) options.ctx.render()
      })
    }, 3000)
    const cleanupName = "events-refresh-interval"
    const cleanup = () => {
      cancelled = true
      clearInterval(interval)
    }
    registerCleanup(cleanupName, cleanup)
    onCleanup(() => {
      cleanup()
      unregisterCleanup(cleanupName)
    })
  })
}
