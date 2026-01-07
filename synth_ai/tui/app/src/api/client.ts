/**
 * HTTP API client for backend communication.
 */

import { appState, backendConfigs, backendKeys } from "../state/app-state"
import type { BackendId } from "../types"

// Type declaration for Node.js process (available at runtime)
declare const process: {
  env: Record<string, string | undefined>
}

/** Ensure URL ends with /api */
function ensureApiBase(url: string): string {
  let base = (url ?? "").trim().replace(/\/+$/, "")
  if (!base) return ""
  if (!base.endsWith("/api")) {
    base = base + "/api"
  }
  return base
}

export function getBackendConfig(id: BackendId = appState.currentBackend): {
  id: BackendId
  label: string
  baseUrl: string
  baseRoot: string
  apiKey: string
} {
  const config = backendConfigs[id]
  // Use the selected backend's URL (don't override with SYNTH_TUI_API_BASE when backend is explicitly selected)
  // SYNTH_TUI_API_BASE is only used for initial backend selection via launcher, not for runtime overrides
  const baseUrl = config.baseUrl
  // For local/dev backends, check SYNTH_API_KEY if key is empty
  let apiKey = backendKeys[id]
  if ((id === "local" || id === "dev") && (!apiKey || !apiKey.trim())) {
    const envKey = id === "local"
      ? (process.env.SYNTH_API_KEY || process.env.SYNTH_TUI_API_KEY_LOCAL || "")
      : (process.env.SYNTH_API_KEY || process.env.SYNTH_TUI_API_KEY_DEV || "")
    apiKey = envKey
  }
  return {
    id,
    label: config.label,
    baseUrl,
    baseRoot: baseUrl.replace(/\/api$/, ""),
    apiKey,
  }
}

export function getActiveApiKey(): string {
  return getBackendConfig().apiKey
}

export function getActiveBaseUrl(): string {
  return getBackendConfig().baseUrl
}

export function getActiveBaseRoot(): string {
  return getBackendConfig().baseRoot
}

export async function apiGet(path: string): Promise<any> {
  const { baseUrl, apiKey, label } = getBackendConfig()
  if (!apiKey) {
    throw new Error(`Missing API key for ${label}`)
  }
  const res = await fetch(`${baseUrl}${path}`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    const suffix = body ? ` - ${body.slice(0, 160)}` : ""
    throw new Error(`GET ${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json()
}

export async function apiGetV1(path: string): Promise<any> {
  const { baseRoot, apiKey, label } = getBackendConfig()
  if (!apiKey) {
    throw new Error(`Missing API key for ${label}`)
  }
  const res = await fetch(`${baseRoot}/api/v1${path}`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    const suffix = body ? ` - ${body.slice(0, 160)}` : ""
    throw new Error(`GET /api/v1${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json()
}

export async function apiPost(path: string, body: any): Promise<any> {
  const { baseUrl, apiKey, label } = getBackendConfig()
  if (!apiKey) {
    throw new Error(`Missing API key for ${label}`)
  }
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    const suffix = text ? ` - ${text.slice(0, 160)}` : ""
    throw new Error(`POST ${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json().catch(() => ({}))
}

export async function refreshHealth(): Promise<string> {
  try {
    // Use current backend configuration (reads appState.currentBackend)
    const baseRoot = getActiveBaseRoot()
    const res = await fetch(`${baseRoot}/health`)
    return res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    return `err(${err?.message || "unknown"})`
  }
}
