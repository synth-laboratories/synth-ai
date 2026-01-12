import { getRequestSignal, mergeAbortSignals } from "./request"

export type SseMessage = {
  event?: string
  data?: string
  id?: string
}

export type SseConnection = {
  disconnect: () => void
}

export function connectSse(
  url: string,
  options: {
    headers?: HeadersInit
    signal?: AbortSignal
    includeScope?: boolean
    onMessage: (message: SseMessage) => void
    onError?: (error: Error) => void
    onOpen?: () => void
  },
): SseConnection {
  let aborted = false
  const controller = new AbortController()
  const baseSignal = getRequestSignal({
    signal: options.signal,
    includeScope: options.includeScope ?? false,
  })
  const signal = mergeAbortSignals([baseSignal, controller.signal]) ?? baseSignal

  void (async () => {
    try {
      const res = await fetch(url, {
        headers: options.headers,
        signal,
      })

      if (!res.ok) {
        const body = await res.text().catch(() => "")
        throw new Error(`SSE stream failed: HTTP ${res.status} ${res.statusText} - ${body.slice(0, 100)}`)
      }

      if (!res.body) {
        throw new Error("SSE stream: no response body")
      }

      options.onOpen?.()

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let currentEvent: SseMessage = {}

      while (!aborted) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          if (line.startsWith(":")) {
            continue
          }

          if (line === "") {
            if (currentEvent.data) {
              options.onMessage(currentEvent)
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
      }
    } catch (err) {
      if (aborted || (err as { name?: string })?.name === "AbortError") return
      options.onError?.(err instanceof Error ? err : new Error(String(err)))
    }
  })()

  return {
    disconnect: () => {
      aborted = true
      controller.abort()
    },
  }
}
