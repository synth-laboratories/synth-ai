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
): Promise<void> {
  const { setUi } = ctx

  try {
    const health = await checkBackendHealth()
    setUi("healthStatus", health)
  } catch (err: any) {
    if (isAbortError(err)) return
    setUi("healthStatus", `err(${err?.message || "unknown"})`)
  }
}
