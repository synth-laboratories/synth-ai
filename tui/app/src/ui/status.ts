/**
 * Status bar text formatting.
 */
import type { AppContext } from "../context"

export function formatStatus(ctx: AppContext): string {
	const { snapshot, appState } = ctx.state
	const balance = snapshot.balanceDollars == null
		? "-"
		: `$${snapshot.balanceDollars.toFixed(2)}`
	const ts = snapshot.lastRefresh
		? new Date(snapshot.lastRefresh).toLocaleTimeString()
		: "-"
	const health = `health=${appState.healthStatus}`
	if (snapshot.lastError) {
		return `Balance: ${balance} | Last refresh: ${ts} | ${health} | Error: ${snapshot.lastError}`
	}
	return `Balance: ${balance} | Last refresh: ${ts} | ${health} | ${snapshot.status}`
}
