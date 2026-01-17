/**
 * Identity (user/org) + balance fetching.
 */
import type { AppContext } from "../context"
import { apiGet, checkBackendHealth, AuthenticationError } from "./client"
import { isAbortError } from "../utils/abort"
import { isAborted } from "../utils/request"

export type RefreshIdentityResult = {
  authError: boolean
}

const HEALTH_COOLDOWN_MS = (() => {
  const rawSeconds = parseFloat(process.env.SYNTH_TUI_HEALTH_COOLDOWN || "60")
  const seconds = Number.isFinite(rawSeconds) ? rawSeconds : 60
  return Math.max(0, seconds) * 1000
})()

const healthState = {
  lastCheckAt: 0,
  inFlight: false,
}

export async function refreshIdentity(
  ctx: AppContext,
): Promise<RefreshIdentityResult> {
  const { data } = ctx.state
  const { setData } = ctx

  try {
    const me = await apiGet("/me", { version: "v1" })
    setData("orgId", typeof me?.org_id === "string" ? me.org_id : null)
    setData("userId", typeof me?.user_id === "string" ? me.user_id : null)
    setData("orgName", typeof me?.org_name === "string" ? me.org_name : null)
    setData("userEmail", typeof me?.user_email === "string" ? me.user_email : null)
  } catch (err: any) {
    if (err instanceof AuthenticationError) {
      // API key is invalid - clear it and signal re-auth needed
      process.env.SYNTH_API_KEY = ""
      return { authError: true }
    }
    if (isAbortError(err)) return { authError: false }
    setData("orgId", data.orgId || null)
    setData("userId", data.userId || null)
    setData("orgName", data.orgName || null)
    setData("userEmail", data.userEmail || null)
  }

  // Get balance from Autumn via backend proxy
  // Backend returns raw Autumn customer response with entitlements array
  // We need the "usage" entitlement with interval="lifetime" for the actual balance
  try {
    if (isAborted()) return { authError: false }
    const autumnBalance = await apiGet("/balance/autumn-current", { version: "v1" })
    const raw = autumnBalance?.raw
    const entitlements = raw?.entitlements
    let balance: number | null = null
    if (Array.isArray(entitlements)) {
      // Find the usage entitlement with lifetime interval (that's where the balance is)
      const usageEnt = entitlements.find(
        (e: any) => e.feature_id === "usage" && e.interval === "lifetime"
      )
      if (usageEnt && typeof usageEnt.balance === "number") {
        balance = usageEnt.balance
      }
    }
    setData("balanceDollars", balance)
  } catch (err: any) {
    if (isAbortError(err)) return { authError: false }
    setData("balanceDollars", null)
  }

  return { authError: false }
}

export async function refreshHealth(
  ctx: AppContext,
  options: { force?: boolean; cooldownMs?: number } = {},
): Promise<void> {
  const { setUi } = ctx
  const force = options.force === true
  const cooldownMs = Math.max(0, options.cooldownMs ?? HEALTH_COOLDOWN_MS)
  const now = Date.now()

  if (healthState.inFlight) {
    return
  }
  if (!force && now - healthState.lastCheckAt < cooldownMs) {
    return
  }
  healthState.inFlight = true
  let shouldRecord = false

  try {
    const health = await checkBackendHealth()
    setUi("healthStatus", health)
    shouldRecord = true
  } catch (err: any) {
    if (isAbortError(err)) return
    setUi("healthStatus", `err(${err?.message || "unknown"})`)
    shouldRecord = true
  } finally {
    healthState.inFlight = false
    if (shouldRecord) {
      healthState.lastCheckAt = Date.now()
    }
  }
}
