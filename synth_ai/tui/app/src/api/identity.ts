/**
 * Identity (user/org) + balance fetching.
 */
import type { AppContext } from "../context"
import { apiGetV1 } from "./client"

export async function refreshIdentity(ctx: AppContext): Promise<void> {
  const { snapshot } = ctx.state

  try {
    const me = await apiGetV1("/me")
    snapshot.orgId = typeof me?.org_id === "string" ? me.org_id : null
    snapshot.userId = typeof me?.user_id === "string" ? me.user_id : null
  } catch {
    snapshot.orgId = snapshot.orgId || null
    snapshot.userId = snapshot.userId || null
  }

  // Get balance from Autumn via backend proxy
  // Backend returns raw Autumn customer response with entitlements array
  // We need the "usage" entitlement with interval="lifetime" for the actual balance
  try {
    const autumnBalance = await apiGetV1("/balance/autumn-current")
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
  } catch {
    snapshot.balanceDollars = null
  }
}

export async function refreshHealth(ctx: AppContext): Promise<void> {
  const { appState } = ctx.state

  try {
    const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/health`)
    appState.healthStatus = res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    appState.healthStatus = `err(${err?.message || "unknown"})`
  }
}

