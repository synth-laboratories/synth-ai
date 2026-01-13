/**
 * Identity (user/org) + balance fetching.
 */
import type { AppContext } from "../context"
import { apiGet, checkBackendHealth } from "./client"
import { isAbortError } from "../utils/abort"
import { isAborted } from "../utils/request"

export async function refreshIdentity(
  ctx: AppContext,
): Promise<void> {
  const { snapshot } = ctx.state

  try {
    const me = await apiGet("/me", { version: "v1" })
    snapshot.orgId = typeof me?.org_id === "string" ? me.org_id : null
    snapshot.userId = typeof me?.user_id === "string" ? me.user_id : null
    snapshot.orgName = typeof me?.org_name === "string" ? me.org_name : null
    snapshot.userEmail = typeof me?.user_email === "string" ? me.user_email : null
  } catch (err: any) {
    if (isAbortError(err)) return
    snapshot.orgId = snapshot.orgId || null
    snapshot.userId = snapshot.userId || null
    snapshot.orgName = snapshot.orgName || null
    snapshot.userEmail = snapshot.userEmail || null
  }

  // Get balance from Autumn via backend proxy
  // Backend returns raw Autumn customer response with entitlements array
  // We need the "usage" entitlement with interval="lifetime" for the actual balance
  try {
    if (isAborted()) return
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
    snapshot.balanceDollars = balance
  } catch (err: any) {
    if (isAbortError(err)) return
    snapshot.balanceDollars = null
  }
}

export async function refreshHealth(
  ctx: AppContext,
): Promise<void> {
  const { appState } = ctx.state

  try {
    appState.healthStatus = await checkBackendHealth()
  } catch (err: any) {
    if (isAbortError(err)) return
    appState.healthStatus = `err(${err?.message || "unknown"})`
  }
}
