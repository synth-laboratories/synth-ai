/**
 * Header / status / footer text formatting.
 */
import type { AppContext } from "../context"
import { getBackendConfig, getActiveBaseRoot } from "../api/client"

export function formatHeaderMeta(ctx: AppContext): string {
  const { snapshot } = ctx.state
  const org = snapshot.orgId || "-"
  const user = snapshot.userId || "-"
  const balance = snapshot.balanceDollars == null ? "unknown" : `$${snapshot.balanceDollars.toFixed(2)}`
  const backendLabel = getBackendConfig().label
  return `backend: ${backendLabel}  org: ${org}  user: ${user}  Balance: ${balance}`
}

export function formatStatus(ctx: AppContext): string {
  const { snapshot, appState } = ctx.state
  const ts = snapshot.lastRefresh ? new Date(snapshot.lastRefresh).toLocaleTimeString() : "-"
  const baseLabel = getActiveBaseRoot().replace(/^https?:\/\//, "")
  const health = `health=${appState.healthStatus}`
  if (snapshot.lastError) {
    return `Last refresh: ${ts} | ${health} | ${baseLabel} | Error: ${snapshot.lastError}`
  }
  return `Last refresh: ${ts} | ${health} | ${baseLabel} | ${snapshot.status}`
}

export function footerText(ctx: AppContext): string {
  const { appState } = ctx.state

  // OpenCode view - typing mode, escape to exit
  if (appState.principalPane === "opencode") {
    const sessionInfo = appState.openCodeSessionId
      ? `session=${appState.openCodeSessionId.slice(-8)}`
      : "no session"
    return `Keys: esc back | ↑↓ scroll | enter send | ctrl+o sessions | ctrl+c quit | ${sessionInfo}`
  }

  // Jobs view keybindings
  const filterLabel = appState.eventFilter ? `filter=${appState.eventFilter}` : "filter=off"
  const jobFilterLabel = appState.jobStatusFilter.size
    ? `status=${Array.from(appState.jobStatusFilter).join(",")}`
    : "status=all"
  return `Keys: g opencode | b jobs | tab toggle | j/k nav | enter view | r refresh | o sessions | l login | t settings | f ${filterLabel} | J ${jobFilterLabel} | u tunnels | c cancel | d usage | q quit`
}




