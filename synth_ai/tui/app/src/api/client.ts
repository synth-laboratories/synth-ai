/**
 * HTTP API client for backend communication.
 *
 * URLs come from process.env (seeded by launcher.py, then updated on mode switch).
 * API key comes from process.env.SYNTH_API_KEY.
 */

import { fetchWithTimeout } from "../utils/request"
import { log } from "../utils/log"
import { DEFAULT_API_TIMEOUT_MS, DEFAULT_HEALTH_TIMEOUT_MS } from "../network"

/**
 * Thrown when API returns 401 (invalid/expired/revoked API key).
 * Used to trigger re-authentication flow.
 */
export class AuthenticationError extends Error {
  constructor(message: string, public status: number) {
    super(message)
    this.name = "AuthenticationError"
  }
}

function sanitizeErrorBody(text: string, maxLen: number): string {
  const raw = (text ?? "").toString()
  if (!raw) return ""

  // Strip HTML tags and collapse whitespace/control chars.
  const noTags = raw.replace(/<[^>]+>/g, " ")
  const collapsed = noTags.replace(/[\r\n\t]+/g, " ").replace(/\s+/g, " ").trim()
  const safe = collapsed.replace(/[^\x20-\x7E]/g, "") // keep printable ASCII
  return safe.length > maxLen ? safe.slice(0, maxLen) : safe
}

async function parseJsonOrThrow(res: Response, label: string): Promise<any> {
  const contentType = res.headers.get("content-type") || ""
  const isJson = contentType.includes("application/json") || contentType.includes("application/problem+json")
  const startRead = Date.now()
  const text = await res.text().catch(() => "")
  const readMs = Date.now() - startRead
  const bytes = Buffer.byteLength(text, "utf8")
  const contentLength = res.headers.get("content-length")

  if (!isJson) {
    const snippet = sanitizeErrorBody(text, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    log("http", "parse.skip", {
      label,
      status: res.status,
      contentType,
      contentLength,
      bytes,
      readMs,
    })
    throw new Error(`${label}: expected JSON but got ${contentType || "unknown content-type"}${suffix}`)
  }

  const startParse = Date.now()
  try {
    const parsed = text ? JSON.parse(text) : {}
    const parseMs = Date.now() - startParse
    log("http", "parse.json", {
      label,
      status: res.status,
      contentType,
      contentLength,
      bytes,
      readMs,
      parseMs,
    })
    return parsed
  } catch {
    const parseMs = Date.now() - startParse
    const snippet = sanitizeErrorBody(text, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    log("http", "parse.error", {
      label,
      status: res.status,
      contentType,
      contentLength,
      bytes,
      readMs,
      parseMs,
    })
    throw new Error(`${label}: invalid JSON response${suffix}`)
  }
}

export type ApiVersion = "v0" | "v1"

export type ApiRequestOptions = {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE"
  body?: unknown
  headers?: Record<string, string>
  signal?: AbortSignal
  version?: ApiVersion
  timeoutMs?: number
}

function getBackendBaseUrl(): string {
  return process.env.SYNTH_BACKEND_URL || ""
}

export function buildApiUrl(path: string, version: ApiVersion = "v0"): string {
  const prefix = version === "v1" ? "/api/v1" : "/api"
  return `${getBackendBaseUrl()}${prefix}${path}`
}

function getApiKey(): string {
  if (!process.env.SYNTH_API_KEY) {
    throw new Error("Missing API key")
  }
  return process.env.SYNTH_API_KEY
}

export function getAuthHeaders(): Record<string, string> {
  return { Authorization: `Bearer ${getApiKey()}` }
}

export async function apiRequest(path: string, options: ApiRequestOptions = {}): Promise<any> {
  const {
    method = "GET",
    body,
    headers,
    signal,
    version = "v0",
    timeoutMs,
  } = options
  const requestHeaders: Record<string, string> = {
    ...getAuthHeaders(),
    ...(headers ?? {}),
  }

  const hasBody = body !== undefined
  const contentType = requestHeaders["Content-Type"]
  const shouldSerializeJson = hasBody && (!contentType || contentType.includes("application/json"))

  if (shouldSerializeJson && !contentType) {
    requestHeaders["Content-Type"] = "application/json"
  }

  const url = buildApiUrl(path, version)
  try {
    const res = await fetchWithTimeout(url, {
      method,
      headers: requestHeaders,
      body: shouldSerializeJson ? JSON.stringify(body) : (body as BodyInit | undefined),
      signal,
      timeoutMs: timeoutMs ?? DEFAULT_API_TIMEOUT_MS,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => "")
      const snippet = sanitizeErrorBody(text, 200)
      const suffix = snippet ? ` - ${snippet}` : ""
      const label = version === "v1" ? `${method} /api/v1${path}` : `${method} ${path}`
      // 401 means the API key is invalid/expired/revoked - trigger re-auth
      if (res.status === 401) {
        throw new AuthenticationError(
          `${label}: HTTP ${res.status} ${res.statusText}${suffix}`,
          res.status
        )
      }
      throw new Error(`${label}: HTTP ${res.status} ${res.statusText}${suffix}`)
    }
    const label = version === "v1" ? `${method} /api/v1${path}` : `${method} ${path}`
    return await parseJsonOrThrow(res, label)
  } catch (err: any) {
    throw err
  }
}

export async function apiGet(path: string, options: ApiRequestOptions = {}): Promise<any> {
  return await apiRequest(path, { ...options, method: "GET" })
}

export async function apiPost(path: string, body: unknown, options: ApiRequestOptions = {}): Promise<any> {
  return await apiRequest(path, { ...options, method: "POST", body })
}

export async function checkBackendHealth(options: { signal?: AbortSignal } = {}): Promise<string> {
  const { signal } = options
  const url = `${getBackendBaseUrl()}/api/health`
  try {
    const res = await fetchWithTimeout(url, {
      signal,
      includeScope: false,
      timeoutMs: DEFAULT_HEALTH_TIMEOUT_MS,
    })
    return res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    return `err(${err?.message || "unknown"})`
  }
}
