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

  try {
    const balance = await apiGetV1("/balance/autumn-normalized")
    const cents = balance?.remaining_credits_cents
    const dollars = typeof cents === "number" && Number.isFinite(cents) ? cents / 100 : null
    snapshot.balanceDollars = dollars
  } catch {
    snapshot.balanceDollars = snapshot.balanceDollars || null
  }
}

export async function refreshHealth(ctx: AppContext): Promise<void> {
  const { appState } = ctx.state

  try {
    const { getActiveBaseRoot } = await import("./client")
    const res = await fetch(`${getActiveBaseRoot()}/health`)
    appState.healthStatus = res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    appState.healthStatus = `err(${err?.message || "unknown"})`
  }
}

