/**
 * Device code authentication flow for TUI.
 *
 * Frontend URL is determined by the current backend mode:
 * - prod: https://usesynth.ai
 * - dev/local: http://localhost:3000
 */

import { spawn } from "node:child_process"
import { appState, getFrontendUrl } from "./state/app-state"

export type AuthSession = {
  deviceCode: string
  verificationUri: string
  expiresAt: number
}

export type AuthResult = {
  success: boolean
  apiKey: string | null
  error: string | null
}

export type AuthStatus =
  | { state: "idle" }
  | { state: "initializing" }
  | { state: "waiting"; verificationUri: string }
  | { state: "polling" }
  | { state: "success"; apiKey: string }
  | { state: "error"; message: string }

const POLL_INTERVAL_MS = 3000

/** Get the current frontend URL based on backend mode */
function getAuthFrontendUrl(): string {
  return getFrontendUrl(appState.currentBackend)
}

/**
 * Initialize a handshake session.
 */
export async function initAuthSession(): Promise<AuthSession> {
  const frontendUrl = getAuthFrontendUrl()
  const initUrl = `${frontendUrl}/api/sdk/handshake/init`

  const res = await fetch(initUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  })

  if (!res.ok) {
    const body = await res.text().catch(() => "")
    throw new Error(`Handshake init failed (${res.status}): ${body || "no response"}`)
  }

  const data = await res.json()
  const deviceCode = String(data.device_code || "").trim()
  const verificationUri = String(data.verification_uri || "").trim()
  const expiresIn = Number(data.expires_in) || 600

  if (!deviceCode || !verificationUri) {
    throw new Error("Handshake init response missing device_code or verification_uri")
  }

  return {
    deviceCode,
    verificationUri,
    expiresAt: Date.now() + expiresIn * 1000,
  }
}

/**
 * Poll for token exchange completion.
 */
export async function pollForToken(
  deviceCode: string,
): Promise<{ apiKey: string | null; expired: boolean; error: string | null }> {
  const frontendUrl = getAuthFrontendUrl()
  const tokenUrl = `${frontendUrl}/api/sdk/handshake/token`

  try {
    const res = await fetch(tokenUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_code: deviceCode }),
    })

    if (res.status === 428) {
      return { apiKey: null, expired: false, error: null }
    }

    if (res.status === 404 || res.status === 410) {
      return { apiKey: null, expired: true, error: "Device code expired" }
    }

    if (!res.ok) {
      const body = await res.text().catch(() => "")
      return { apiKey: null, expired: false, error: `Token exchange failed: ${body}` }
    }

    const data = await res.json()
    const keys = data.keys || {}
    const synthKey = String(keys.synth || "").trim()

    if (!synthKey) {
      return { apiKey: null, expired: false, error: "No API key in response" }
    }

    return { apiKey: synthKey, expired: false, error: null }
  } catch (err: any) {
    return { apiKey: null, expired: false, error: err?.message || "Network error" }
  }
}

/**
 * Open a URL in the default browser.
 */
export function openBrowser(url: string): void {
  const platform = process.platform
  let cmd: string
  let args: string[]

  if (platform === "darwin") {
    cmd = "open"
    args = [url]
  } else if (platform === "win32") {
    cmd = "cmd"
    args = ["/c", "start", "", url]
  } else {
    cmd = "xdg-open"
    args = [url]
  }

  try {
    const child = spawn(cmd, args, {
      detached: true,
      stdio: "ignore",
    })
    child.unref()
  } catch {
    // ignore
  }
}

/**
 * Run the full device code authentication flow.
 */
export async function runDeviceCodeAuth(
  onStatus?: (status: AuthStatus) => void,
): Promise<AuthResult> {
  const updateStatus = (status: AuthStatus) => {
    if (onStatus) onStatus(status)
  }

  try {
    updateStatus({ state: "initializing" })
    const session = await initAuthSession()

    updateStatus({ state: "waiting", verificationUri: session.verificationUri })
    openBrowser(session.verificationUri)

    updateStatus({ state: "polling" })
    while (Date.now() < session.expiresAt) {
      const result = await pollForToken(session.deviceCode)

      if (result.apiKey) {
        updateStatus({ state: "success", apiKey: result.apiKey })
        return { success: true, apiKey: result.apiKey, error: null }
      }

      if (result.expired) {
        updateStatus({ state: "error", message: "Authentication timed out" })
        return { success: false, apiKey: null, error: "Authentication timed out" }
      }

      await sleep(POLL_INTERVAL_MS)
    }

    updateStatus({ state: "error", message: "Authentication timed out" })
    return { success: false, apiKey: null, error: "Authentication timed out" }
  } catch (err: any) {
    const message = err?.message || "Authentication failed"
    updateStatus({ state: "error", message })
    return { success: false, apiKey: null, error: message }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
