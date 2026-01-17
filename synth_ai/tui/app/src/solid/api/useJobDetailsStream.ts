/**
 * SolidJS hook for using the job details SSE stream.
 * Automatically connects when job changes and disconnects on cleanup.
 */
import { createEffect, onCleanup } from "solid-js"
import {
  connectJobDetailsStream,
  type JobDetailsStreamConnection,
  type JobDetailsStreamEvent,
} from "../../api/job-details-stream"
import { registerCleanup, unregisterCleanup } from "../../lifecycle"
import { type SseChannel, setSseConnected } from "../../state/polling"

export interface UseJobDetailsStreamOptions {
  jobId: () => string | null | undefined
  onEvent: (event: JobDetailsStreamEvent) => void
  onError?: (error: Error) => void
  onOpen?: () => void
  sinceSeq?: () => number
  enabled?: () => boolean
  sseKey?: SseChannel
}

/**
 * Hook to subscribe to real-time job details updates.
 * Automatically manages connection lifecycle based on job selection.
 */
export function useJobDetailsStream(options: UseJobDetailsStreamOptions): void {
  let connection: JobDetailsStreamConnection | null = null
  const cleanupName = "job-details-stream"
  const trackSse = (connected: boolean) => {
    if (options.sseKey) {
      setSseConnected(options.sseKey, connected)
    }
  }

  // Cleanup function for disconnecting the stream
  const cleanup = () => {
    if (connection) {
      connection.disconnect()
      connection = null
    }
    trackSse(false)
  }

  createEffect(() => {
    // Disconnect previous stream if any
    cleanup()

    // Check if streaming is enabled
    if (options.enabled && !options.enabled()) {
      return
    }

    const jobId = options.jobId()
    if (!jobId) {
      return
    }

    // Connect to the stream
    connection = connectJobDetailsStream(jobId, options.onEvent, (err) => {
      trackSse(false)
      options.onError?.(err)
    }, () => options.sinceSeq?.() ?? 0, {
      onOpen: () => {
        trackSse(true)
        options.onOpen?.()
      },
    })
    // Re-registering with same name overwrites previous entry (Map semantics)
    registerCleanup(cleanupName, cleanup)
  })

  // Cleanup on component unmount
  onCleanup(() => {
    cleanup()
    unregisterCleanup(cleanupName)
  })
}
