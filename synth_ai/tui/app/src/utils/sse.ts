import { getRequestSignal, mergeAbortSignals, sleep } from "./request"
import { log } from "./log"
import {
  SSE_RECONNECT_BASE_MS,
  SSE_RECONNECT_JITTER_MS,
  SSE_RECONNECT_MAX_MS,
  SSE_RECONNECT_MULTIPLIER,
} from "../network"

export type SseMessage = {
  event?: string
  data?: string
  id?: string
}

export type SseConnection = {
  disconnect: () => void
}

export type SseReconnectOptions = {
  enabled?: boolean
  baseDelayMs?: number
  maxDelayMs?: number
  multiplier?: number
  jitterMs?: number
}

export function connectSse(
  url: string,
  options: {
    headers?: HeadersInit
    signal?: AbortSignal
    includeScope?: boolean
    label?: string
    reconnect?: SseReconnectOptions
    getUrl?: () => string
    onMessage: (message: SseMessage) => void
    onError?: (error: Error) => void
    onOpen?: () => void
  },
): SseConnection {
  let aborted = false
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null
  const controller = new AbortController()

  const baseManaged = getRequestSignal({
    signal: options.signal,
    includeScope: options.includeScope ?? false,
  })
  const mergedManaged = mergeAbortSignals([baseManaged.signal, controller.signal]) ?? baseManaged

  const reconnect = options.reconnect ?? {}
  const reconnectEnabled = reconnect.enabled !== false
  const baseDelay = reconnect.baseDelayMs ?? SSE_RECONNECT_BASE_MS
  const maxDelay = reconnect.maxDelayMs ?? SSE_RECONNECT_MAX_MS
  const multiplier = reconnect.multiplier ?? SSE_RECONNECT_MULTIPLIER
  const jitterMs = reconnect.jitterMs ?? SSE_RECONNECT_JITTER_MS
  const streamLabel = options.label ? `SSE:${options.label}` : "SSE"

  const nextDelay = (attempt: number): number => {
    const exp = baseDelay * Math.pow(multiplier, Math.max(0, attempt - 1))
    const jitter = jitterMs > 0 ? Math.floor(Math.random() * jitterMs) : 0
    return Math.min(maxDelay, Math.floor(exp) + jitter)
  }

  void (async () => {
    let attempt = 0
    try {
      while (!aborted) {
        const targetUrl = options.getUrl ? options.getUrl() : url
        const start = Date.now()
        log("http", `→ GET (${streamLabel}) ${targetUrl}`)
        try {
          const res = await fetch(targetUrl, {
            headers: options.headers,
            signal: mergedManaged.signal,
          })
          log("http", `← ${res.status} GET (${streamLabel}) ${targetUrl} (${Date.now() - start}ms)`)

          if (!res.ok) {
            const body = await res.text().catch(() => "")
            throw new Error(
              `SSE stream failed: HTTP ${res.status} ${res.statusText} - ${body.slice(0, 100)}`,
            )
          }

          if (!res.body) {
            throw new Error("SSE stream: no response body")
          }

          attempt = 0
          options.onOpen?.()

          reader = res.body.getReader()
          const decoder = new TextDecoder()
          let buffer = ""
          let currentEvent: SseMessage = {}
          log("state", "sse reader start", {
            label: streamLabel,
          })

          while (!aborted) {
            const readStart = Date.now()
            const { done, value } = await reader.read()
            const readMs = Date.now() - readStart
            const bytes = value ? value.length : 0
            log("state", "sse read", {
              label: streamLabel,
              done,
              readMs,
              bytes,
              bufferBytes: buffer.length,
            })
            if (done) break
            if (!value || value.length === 0) continue

            const decodeStart = Date.now()
            buffer += decoder.decode(value, { stream: true })
            const decodeMs = Date.now() - decodeStart
            const splitStart = Date.now()
            const lines = buffer.split("\n")
            buffer = lines.pop() ?? ""
            const splitMs = Date.now() - splitStart

            const handleStart = Date.now()
            let messageCount = 0
            let onMessageMs = 0
            let messageBytes = 0

            for (const line of lines) {
              if (line.startsWith(":")) {
                continue
              }

              if (line === "") {
                if (currentEvent.data) {
                  const dataBytes = currentEvent.data.length
                  log("state", "sse message start", {
                    label: streamLabel,
                    dataBytes,
                    event: currentEvent.event ?? null,
                    id: currentEvent.id ?? null,
                  })
                  const messageStart = Date.now()
                  options.onMessage(currentEvent)
                  const messageMs = Date.now() - messageStart
                  onMessageMs += messageMs
                  messageBytes += dataBytes
                  messageCount += 1
                  log("state", "sse message end", {
                    label: streamLabel,
                    dataBytes,
                    event: currentEvent.event ?? null,
                    id: currentEvent.id ?? null,
                    messageMs,
                  })
                }
                currentEvent = {}
                continue
              }

              const colonIdx = line.indexOf(":")
              if (colonIdx === -1) continue

              const field = line.slice(0, colonIdx)
              let value = line.slice(colonIdx + 1)
              if (value.startsWith(" ")) value = value.slice(1)

              switch (field) {
                case "event":
                  currentEvent.event = value
                  break
                case "data":
                  currentEvent.data = (currentEvent.data ?? "") + value
                  break
                case "id":
                  currentEvent.id = value
                  break
              }
            }

            const handleMs = Date.now() - handleStart
            log("state", "sse chunk", {
              label: streamLabel,
              bytes,
              bufferBytes: buffer.length,
              lines: lines.length,
              messages: messageCount,
              messageBytes,
              readMs,
              decodeMs,
              splitMs,
              handleMs,
              onMessageMs,
            })
          }

          if (!aborted) {
            throw new Error("SSE stream closed")
          }
        } catch (err) {
          if (aborted || (err as { name?: string })?.name === "AbortError") return
          log("http", `✗ GET (${streamLabel}) ${targetUrl} - ${(err as Error)?.message}`)
          options.onError?.(err instanceof Error ? err : new Error(String(err)))
        } finally {
          if (reader) {
            await reader.cancel().catch(() => {})
            reader = null
          }
        }

        if (!reconnectEnabled || aborted) break
        attempt += 1
        const delay = nextDelay(attempt)
        try {
          await sleep(delay, { signal: mergedManaged.signal, includeScope: false })
        } catch {
          return
        }
      }
    } finally {
      mergedManaged.dispose()
      baseManaged.dispose()
    }
  })()

  return {
    disconnect: () => {
      aborted = true
      controller.abort()
    },
  }
}
