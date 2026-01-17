import { buildApiUrl, getAuthHeaders } from "./client"
import { connectSse } from "../utils/sse"
import { log } from "../utils/log"

export type JsonStreamConnection = {
  disconnect: () => void
}

type JsonStreamOptions<T> = {
  url?: string
  getUrl?: () => string
  headers?: HeadersInit
  signal?: AbortSignal
  includeScope?: boolean
  label?: string
  onEvent: (event: T) => void
  onError?: (error: Error) => void
  onOpen?: () => void
  parse?: (raw: string) => T | null
}

type ApiJsonStreamOptions<T> = {
  path: string | (() => string)
  headers?: HeadersInit
  signal?: AbortSignal
  includeScope?: boolean
  withAuth?: boolean
  label?: string
  onEvent: (event: T) => void
  onError?: (error: Error) => void
  onOpen?: () => void
  parse?: (raw: string) => T | null
}

function normalizeHeaders(headers?: HeadersInit): Headers {
  const normalized = new Headers(headers)
  if (!normalized.has("Accept")) {
    normalized.set("Accept", "text/event-stream")
  }
  return normalized
}

export function connectJsonStream<T>(options: JsonStreamOptions<T>): JsonStreamConnection {
  const getUrl =
    options.getUrl ??
    (() => {
      if (!options.url) {
        throw new Error("Missing SSE URL")
      }
      return options.url
    })
  const parse = options.parse ?? ((raw: string) => JSON.parse(raw) as T)
  const headers = normalizeHeaders(options.headers)
  const label = options.label ?? "json-stream"

  const connection = connectSse(getUrl(), {
    headers,
    signal: options.signal,
    includeScope: options.includeScope ?? false,
    label: options.label,
    getUrl,
    onOpen: options.onOpen,
    onMessage: (message) => {
      if (!message.data) return
      const dataBytes = message.data.length
      log("state", "stream parse start", { label, dataBytes })
      let parsed: T | null = null
      const parseStart = Date.now()
      try {
        parsed = parse(message.data)
      } catch (err) {
        log("state", "stream parse error", {
          label,
          dataBytes,
          error: err instanceof Error ? err.message : String(err),
        })
        return
      } finally {
        const parseMs = Date.now() - parseStart
        log("state", "stream parse end", { label, dataBytes, parseMs })
      }
      if (parsed != null) {
        const eventStart = Date.now()
        options.onEvent(parsed)
        const eventMs = Date.now() - eventStart
        log("state", "stream event handled", { label, eventMs })
      }
    },
    onError: options.onError,
  })

  return {
    disconnect: () => connection.disconnect(),
  }
}

export function connectApiJsonStream<T>(options: ApiJsonStreamOptions<T>): JsonStreamConnection {
  const getPath = typeof options.path === "function" ? options.path : () => options.path as string
  const getUrl = () => buildApiUrl(getPath())
  const headers = new Headers(options.headers)

  if (options.withAuth !== false) {
    try {
      const authHeaders = getAuthHeaders()
      for (const [key, value] of Object.entries(authHeaders)) {
        headers.set(key, value)
      }
    } catch (err) {
      options.onError?.(err instanceof Error ? err : new Error(String(err)))
      return { disconnect: () => {} }
    }
  }

  return connectJsonStream({
    getUrl,
    headers,
    signal: options.signal,
    includeScope: options.includeScope ?? false,
    label: options.label,
    onOpen: options.onOpen,
    onEvent: options.onEvent,
    onError: options.onError,
    parse: options.parse,
  })
}
