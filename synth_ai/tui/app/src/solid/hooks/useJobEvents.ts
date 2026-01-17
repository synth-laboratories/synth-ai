import { createEffect, createSignal, onCleanup, type Accessor } from "solid-js"

import { refreshEvents } from "../../api/events"
import { refreshHealth } from "../../api/identity"
import type { JobDetailsStreamEvent } from "../../api/job-details-stream"
import type { PrimaryView } from "../../types"
import type { AppContext } from "../../context"
import { registerCleanup, unregisterCleanup } from "../../lifecycle"
import { useJobDetailsStream } from "../api/useJobDetailsStream"
import type { JobEvent } from "../../tui_data"
import { clearEventsTimer, config, pollingState, setPollNextAt, shouldPoll, onSseChange } from "../../state/polling"

type UseJobEventsOptions = {
  ctx: AppContext
  selectedJobId: Accessor<string | null>
  primaryView: Accessor<PrimaryView>
}

export function useJobEvents(options: UseJobEventsOptions): void {
  const { data, ui } = options.ctx.state
  const { setData, setUi } = options.ctx
  const [lastSeenSeq, setLastSeenSeq] = createSignal(0)
  let lastJobId: string | null = null

  createEffect(() => {
    const jobId = options.selectedJobId() ?? null
    if (jobId === lastJobId) return
    lastJobId = jobId
    setLastSeenSeq(0)
  })

  // Subscribe to real-time job details updates via SSE.
  useJobDetailsStream({
    jobId: options.selectedJobId,
    sinceSeq: lastSeenSeq,
    enabled: () => options.primaryView() === "jobs",
    sseKey: "job-events",
    onEvent: (event: JobDetailsStreamEvent) => {
      if (event.seq > lastSeenSeq()) {
        setLastSeenSeq(event.seq)
      }
      // Keep polling cursor in sync with SSE cursor so refreshEvents() doesn't refetch from 0.
      const nextLastSeq = Math.max(ui.lastSeq || 0, event.seq)
      if (nextLastSeq !== ui.lastSeq) {
        setUi("lastSeq", nextLastSeq)
      }

      const jobEvent: JobEvent = {
        seq: event.seq,
        type: event.type,
        message: event.message,
        data: event.data as JobEvent["data"],
        timestamp: new Date(event.ts).toISOString(),
      }

      const existingSeqs = new Set(data.events.map((e) => e.seq))
      if (!existingSeqs.has(jobEvent.seq)) {
        const nextEvents = [...data.events, jobEvent].sort((a, b) => a.seq - b.seq)
        setData("events", nextEvents)
      }
    },
    onError: (error) => {
      // Log but don't show to user - polling will still work as fallback.
      console.error("Job details SSE error:", error.message)
      void refreshHealth(options.ctx)
    },
  })

  // Poll/backfill events to avoid gaps when SSE drops or when selecting an older job.
  createEffect(() => {
    const jobId = options.selectedJobId()
    const enabled = options.primaryView() === "jobs"
    if (!jobId || !enabled) return

    let cancelled = false
    let pollingActive = false
    const refreshOnce = async (): Promise<boolean | null> => {
      if (cancelled) return null
      if (pollingState.eventsInFlight) return null
      if (!shouldPoll("job-events")) return null
      pollingState.eventsInFlight = true
      try {
        const ok = await refreshEvents(options.ctx)
        if (!ok) {
          void refreshHealth(options.ctx)
        }
        return ok
      } catch {
        void refreshHealth(options.ctx)
        return false
      } finally {
        pollingState.eventsInFlight = false
      }
    }

    async function backfillOnce(): Promise<void> {
      if (!shouldPoll("job-events")) return
      // Pull up to ~10 pages (2000 events) max per selection, but stop early if no progress.
      for (let i = 0; i < 10; i++) {
        const beforeSeq = options.ctx.state.ui.lastSeq
        const beforeLen = options.ctx.state.data.events.length
        const ok = await refreshOnce()
        if (ok !== true) break
        if (cancelled) return
        if (
          options.ctx.state.data.events.length === beforeLen &&
          options.ctx.state.ui.lastSeq === beforeSeq
        ) {
          break
        }
      }
    }

    let delayMs = Math.max(0.5, config.eventInterval) * 1000
    const minDelayMs = delayMs
    const maxDelayMs = Math.max(delayMs, Math.max(1, config.maxEventInterval) * 1000)
    const stopPolling = () => {
      pollingActive = false
      clearEventsTimer()
    }
    const schedule = (nextDelay: number) => {
      if (!shouldPoll("job-events")) {
        stopPolling()
        return
      }
      if (pollingState.eventsTimer) clearTimeout(pollingState.eventsTimer)
      const delay = Math.max(0, nextDelay)
      setPollNextAt("job-events", Date.now() + delay)
      pollingState.eventsTimer = setTimeout(() => {
        void run()
      }, delay)
    }

    const run = async () => {
      if (cancelled) return
      if (!shouldPoll("job-events")) {
        stopPolling()
        return
      }
      const ok = await refreshOnce()
      if (ok == null) {
        schedule(delayMs)
        return
      }
      if (ok) {
        delayMs = minDelayMs
      } else {
        delayMs = Math.min(maxDelayMs, Math.floor(delayMs * 1.7))
      }
      schedule(delayMs)
    }

    const startPolling = () => {
      if (cancelled || pollingActive) return
      if (!shouldPoll("job-events")) return
      pollingActive = true
      void (async () => {
        await backfillOnce()
        if (cancelled) return
        if (!shouldPoll("job-events")) {
          pollingActive = false
          return
        }
        schedule(delayMs)
      })()
    }

    const unsubscribeSse = onSseChange("job-events", (connected) => {
      if (connected) {
        stopPolling()
        return
      }
      delayMs = minDelayMs
      startPolling()
    })
    if (shouldPoll("job-events")) {
      startPolling()
    }
    const cleanupName = "events-refresh-interval"
    const cleanup = () => {
      cancelled = true
      stopPolling()
      pollingState.eventsInFlight = false
      unsubscribeSse()
    }
    registerCleanup(cleanupName, cleanup)
    onCleanup(() => {
      cleanup()
      unregisterCleanup(cleanupName)
    })
  })
}
