/**
 * Identity (user/org) + balance fetching.
 */
import type { AppContext } from "../context"
import { apiGetV1 } from "./client"
import { isAbortError } from "../utils/abort"

export async function refreshIdentity(
  ctx: AppContext,
  options: { signal?: AbortSignal } = {},
): Promise<void> {
  const { snapshot } = ctx.state

  try {
    const me = await apiGetV1("/me", options)
    snapshot.orgId = typeof me?.org_id === "string" ? me.org_id : null
    snapshot.userId = typeof me?.user_id === "string" ? me.user_id : null
  } catch (err: any) {
    if (isAbortError(err)) return
    snapshot.orgId = snapshot.orgId || null
    snapshot.userId = snapshot.userId || null
  }

  // Get balance from Autumn via backend proxy
  // Backend returns raw Autumn customer response with entitlements array
  // We need the "usage" entitlement with interval="lifetime" for the actual balance
  try {
    if (options.signal?.aborted) return
    const autumnBalance = await apiGetV1("/balance/autumn-current", options)
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
  options: { signal?: AbortSignal } = {},
): Promise<void> {
  const { appState } = ctx.state

  try {
    const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/health`, {
      signal: options.signal,
    })
    appState.healthStatus = res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    if (isAbortError(err)) return
    appState.healthStatus = `err(${err?.message || "unknown"})`
  }
}
