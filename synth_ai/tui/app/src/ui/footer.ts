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

	const keys = [
		"e events",
		"b jobs",
		"tab toggle",
		"j/k nav",
		"enter view",
		"r refresh",
		"l logout",
		`f ${filterLabel}`,
		`shift+j ${jobFilterLabel}`,
		"c cancel",
		"a artifacts",
		"o results",
		"s snapshot",
		...(process.env.SYNTH_API_KEY ? ["p profile"] : []),
		"q quit",
	]

	return `Keys: ${keys.join(" | ")}`
}
