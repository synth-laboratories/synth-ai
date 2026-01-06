/**
 * Header / status / footer text formatting.
 */
import type { AppContext } from "../context"

export function formatStatus(ctx: AppContext): string {
  const { snapshot, appState } = ctx.state
  const balance = snapshot.balanceDollars == null ? "-" : `$${snapshot.balanceDollars.toFixed(2)}`
  const ts = snapshot.lastRefresh ? new Date(snapshot.lastRefresh).toLocaleTimeString() : "-"
  const health = `health=${appState.healthStatus}`
  if (snapshot.lastError) {
    return `Balance: ${balance} | Last refresh: ${ts} | ${health} | Error: ${snapshot.lastError}`
  }
  return `Balance: ${balance} | Last refresh: ${ts} | ${health} | ${snapshot.status}`
}

export function footerText(ctx: AppContext): string {
  const { appState } = ctx.state
  const filterLabel = appState.eventFilter ? `filter=${appState.eventFilter}` : "filter=off"
  const jobFilterLabel = appState.jobStatusFilter.size
    ? `status=${Array.from(appState.jobStatusFilter).join(",")}`
    : "status=all"
  const profileKey = process.env.SYNTH_API_KEY ? " | p profile" : ""
  return `Keys: e events | b jobs | tab toggle | j/k nav | enter view | r refresh | l login | L logout | f ${filterLabel} | shift+j ${jobFilterLabel} | c cancel | a artifacts | o results | s snapshot${profileKey} | q quit`
}




