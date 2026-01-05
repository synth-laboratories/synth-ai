/**
 * Device code authentication flow for TUI.
 * 
 * Port of synth_ai/core/auth.py to TypeScript.
 * Uses the existing frontend handshake endpoints.
 */

import { spawn } from "node:child_process"

export type BackendId = "prod" | "dev" | "local"

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

/**
 * Get the frontend URL for a given backend.
 * The frontend is where the handshake endpoints live.
 */
function getFrontendUrl(backend: BackendId): string {
  switch (backend) {
    case "prod":
      return process.env.SYNTH_TUI_FRONTEND_PROD || "https://www.usesynth.ai"
    case "dev":
      return process.env.SYNTH_TUI_FRONTEND_DEV || "https://synth-frontend-dev.onrender.com"
    case "local":
      return process.env.SYNTH_TUI_FRONTEND_LOCAL || "http://localhost:3000"
  }
}

/**
 * Initialize a handshake session.
 * Returns device_code and verification_uri.
 */
export async function initAuthSession(backend: BackendId): Promise<AuthSession> {
  const frontendUrl = getFrontendUrl(backend)
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
 * Returns API key when user completes auth, or null if still pending.
 */
export async function pollForToken(
  backend: BackendId,
  deviceCode: string,
): Promise<{ apiKey: string | null; expired: boolean; error: string | null }> {
  const frontendUrl = getFrontendUrl(backend)
  const tokenUrl = `${frontendUrl}/api/sdk/handshake/token`

  try {
    const res = await fetch(tokenUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_code: deviceCode }),
    })

    if (res.status === 428) {
      // authorization_pending - user hasn't completed auth yet
      return { apiKey: null, expired: false, error: null }
    }

    if (res.status === 404 || res.status === 410) {
      // Device code expired or revoked
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
 * Cross-platform support for macOS, Linux, and Windows.
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
    // Linux and others
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
    // Ignore errors - browser open is best-effort
  }
}

/**
 * Run the full device code authentication flow.
 * 
 * @param backend - Which backend to authenticate against
 * @param onStatus - Callback for status updates (for UI)
 * @returns AuthResult with API key on success
 */
export async function runDeviceCodeAuth(
  backend: BackendId,
  onStatus?: (status: AuthStatus) => void,
): Promise<AuthResult> {
  const updateStatus = (status: AuthStatus) => {
    if (onStatus) onStatus(status)
  }

  try {
    // Initialize handshake
    updateStatus({ state: "initializing" })
    const session = await initAuthSession(backend)

    // Open browser
    updateStatus({ state: "waiting", verificationUri: session.verificationUri })
    openBrowser(session.verificationUri)

    // Poll for completion
    updateStatus({ state: "polling" })
    while (Date.now() < session.expiresAt) {
      const result = await pollForToken(backend, session.deviceCode)

      if (result.apiKey) {
        updateStatus({ state: "success", apiKey: result.apiKey })
        return { success: true, apiKey: result.apiKey, error: null }
      }

      if (result.expired) {
        updateStatus({ state: "error", message: "Authentication timed out" })
        return { success: false, apiKey: null, error: "Authentication timed out" }
      }

      if (result.error) {
        // Transient error, keep polling
      }

      await sleep(POLL_INTERVAL_MS)
    }

    // Timeout
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

