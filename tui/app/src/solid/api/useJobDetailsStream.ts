/**
 * SolidJS hook for using the job details SSE stream.
 * Automatically connects when job changes and disconnects on cleanup.
 */
import { createEffect, onCleanup } from "solid-js"
import {
  connectJobDetailsStream,
  type JobDetailsStreamConnection,
  type JobDetailsStreamEvent,
} from "./job-details-stream"
import { registerCleanup, unregisterCleanup } from "../../lifecycle"

export interface UseJobDetailsStreamOptions {
  jobId: () => string | null | undefined
  onEvent: (event: JobDetailsStreamEvent) => void
  onError?: (error: Error) => void
  sinceSeq?: () => number
  enabled?: () => boolean
}

/**
 * Hook to subscribe to real-time job details updates.
 * Automatically manages connection lifecycle based on job selection.
 */
export function useJobDetailsStream(options: UseJobDetailsStreamOptions): void {
  let connection: JobDetailsStreamConnection | null = null
  const cleanupName = "job-details-stream"

  // Cleanup function for disconnecting the stream
  const cleanup = () => {
    if (connection) {
      connection.disconnect()
      connection = null
    }
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

    const sinceSeq = options.sinceSeq?.() ?? 0

    // Connect to the stream
    connection = connectJobDetailsStream(
      jobId,
      options.onEvent,
      options.onError,
      sinceSeq,
    )
    // Re-registering with same name overwrites previous entry (Map semantics)
    registerCleanup(cleanupName, cleanup)
  })

  // Cleanup on component unmount
  onCleanup(() => {
    cleanup()
    unregisterCleanup(cleanupName)
  })
}

