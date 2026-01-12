/**
 * HTTP API client for backend communication.
 *
 * URLs come from launcher.py (which gets them from urls.py).
 * API key comes from process.env.SYNTH_API_KEY.
 */

import { getRequestSignal } from "../utils/request"

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
  const text = await res.text().catch(() => "")

  if (!isJson) {
    const snippet = sanitizeErrorBody(text, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    throw new Error(`${label}: expected JSON but got ${contentType || "unknown content-type"}${suffix}`)
  }

  try {
    return text ? JSON.parse(text) : {}
  } catch {
    const snippet = sanitizeErrorBody(text, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
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
  } = options
  const requestHeaders: Record<string, string> = {
    ...getAuthHeaders(),
    ...(headers ?? {}),
  }

  const hasBody = body !== undefined
  if (hasBody && !requestHeaders["Content-Type"]) {
    requestHeaders["Content-Type"] = "application/json"
  }

  const res = await fetch(buildApiUrl(path, version), {
    method,
    headers: requestHeaders,
    body: hasBody ? JSON.stringify(body) : undefined,
    signal: getRequestSignal({ signal }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    const snippet = sanitizeErrorBody(text, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    const label = version === "v1" ? `${method} /api/v1${path}` : `${method} ${path}`
    throw new Error(`${label}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  const label = version === "v1" ? `${method} /api/v1${path}` : `${method} ${path}`
  return await parseJsonOrThrow(res, label)
}

export async function apiGet(path: string, options: ApiRequestOptions = {}): Promise<any> {
  return await apiRequest(path, { ...options, method: "GET" })
}

export async function apiPost(path: string, body: unknown, options: ApiRequestOptions = {}): Promise<any> {
  return await apiRequest(path, { ...options, method: "POST", body })
}

export async function checkBackendHealth(options: { signal?: AbortSignal } = {}): Promise<string> {
  const { signal } = options
  try {
    const res = await fetch(`${getBackendBaseUrl()}/health`, {
      signal: getRequestSignal({ signal, includeScope: false }),
    })
    return res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    return `err(${err?.message || "unknown"})`
  }
}
