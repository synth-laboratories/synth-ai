/**
 * Header / status / footer text formatting.
 */
import type { AppContext } from "../context"

export function formatHeaderMeta(ctx: AppContext): string {
  const { snapshot } = ctx.state
  const org = snapshot.orgId || "-"
  const user = snapshot.userId || "-"
  const balance = snapshot.balanceDollars == null ? "-" : `$${snapshot.balanceDollars.toFixed(2)}`
  return `org: ${org}  user: ${user}  balance: ${balance}`
}

export function formatStatus(ctx: AppContext): string {
  const { snapshot, appState } = ctx.state
  const ts = snapshot.lastRefresh ? new Date(snapshot.lastRefresh).toLocaleTimeString() : "-"
  const baseLabel = (process.env.SYNTH_BACKEND_URL || "").replace(/^https?:\/\//, "")
  const health = `health=${appState.healthStatus}`
  if (snapshot.lastError) {
    return `Last refresh: ${ts} | ${health} | ${baseLabel} | Error: ${snapshot.lastError}`
  }
  return `Last refresh: ${ts} | ${health} | ${baseLabel} | ${snapshot.status}`
}

export function footerText(ctx: AppContext): string {
  const { appState } = ctx.state
  const filterLabel = appState.eventFilter ? `filter=${appState.eventFilter}` : "filter=off"
  const jobFilterLabel = appState.jobStatusFilter.size
    ? `status=${Array.from(appState.jobStatusFilter).join(",")}`
    : "status=all"
  return `Keys: e events | b jobs | tab toggle | j/k nav | enter view | r refresh | l login | L logout | f ${filterLabel} | shift+j ${jobFilterLabel} | c cancel | a artifacts | s snapshot | q quit`
}




