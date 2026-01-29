/**
 * Footer keyboard shortcuts ribbon.
 */
import type { AppContext } from "../context"

export function footerText(ctx: AppContext): string {
	const { appState } = ctx.state
	const filterLabel = appState.eventFilter ? `filter=${appState.eventFilter}` : "filter=off"
	const jobFilterLabel = appState.jobStatusFilter.size
		? `status=${Array.from(appState.jobStatusFilter).join(",")}`
		: "status=all"

	// Show different keys based on active pane
	if (appState.activePane === "logs") {
		const keys = [
			"j/k select",
			"enter open",
			"e events",
			"b jobs",
			"tab toggle",
			"n create",
			"q quit",
		].filter(Boolean)
		return `Keys: ${keys.join(" | ")}`
	}

	const keys = [
		"e events",
		"b jobs",
		"g logs",
		"n create",
		"tab toggle",
		"j/k nav",
		"enter view",
		"r refresh",
		"l logout",
		`f ${filterLabel}`,
		`shift+j ${jobFilterLabel}`,
		"c cancel",
		"a artifacts",
		"v candidates",
		"s snapshot",
		...(process.env.SYNTH_API_KEY ? ["p profile"] : []),
		"q quit",
	]

	return `Keys: ${keys.join(" | ")}`
}
