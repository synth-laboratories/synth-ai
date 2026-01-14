import { buildApiUrl, getAuthHeaders } from "./client"
import { connectSse } from "../utils/sse"

export type JsonStreamConnection = {
  disconnect: () => void
}

type JsonStreamOptions<T> = {
  url?: string
  getUrl?: () => string
  headers?: HeadersInit
  signal?: AbortSignal
  includeScope?: boolean
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

  const connection = connectSse(getUrl(), {
    headers,
    signal: options.signal,
    includeScope: options.includeScope ?? false,
    getUrl,
    onOpen: options.onOpen,
    onMessage: (message) => {
      if (!message.data) return
      try {
        const parsed = parse(message.data)
        if (parsed != null) {
          options.onEvent(parsed)
        }
      } catch {
        // Ignore parse errors.
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
    onOpen: options.onOpen,
    onEvent: options.onEvent,
    onError: options.onError,
    parse: options.parse,
  })
}
