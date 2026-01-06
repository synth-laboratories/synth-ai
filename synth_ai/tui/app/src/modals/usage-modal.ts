/**
 * Usage modal controller.
 * Shows plan info, rollout credits, and usage breakdown.
 */
import type { AppContext } from "../context"
import { wrapModalText, clamp, type ModalController } from "./base"
import { apiGetV1 } from "../api/client"

export interface UsageData {
  plan_type: "free" | "pro" | "team" | "byok"
  status: "active" | "cancelled" | "past_due" | "trialing" | "inactive"
  rollout_credits_balance_usd?: number | null
  rollout_credits_used_this_period_usd?: number | null
  limits: {
    monthly_rollout_credits_usd: number
    max_overdraft_usd: number
    unlimited_non_rollout: boolean
    team_features_enabled: boolean
    byok_enabled: boolean
  }
  usage_summary?: {
    total_cost_usd: number
    total_charged_usd: number
    total_uncharged_usd: number  // "Savings"
    by_type: Array<{
      usage_type: string
      total_cost_usd: number
      charged_cost_usd: number
      uncharged_cost_usd: number
      event_count: number
      byok_event_count: number
    }>
  }
}

/**
 * Format plan name with emoji for display.
 */
function formatPlanName(planType: string): string {
  switch (planType) {
    case "pro":
      return "\u2b50 Pro"
    case "team":
      return "\ud83d\ude80 Team"
    case "byok":
      return "\ud83d\udd11 BYOK"
    case "free":
    default:
      return "\ud83c\udd93 Free"
  }
}

/**
 * Format status with indicator.
 */
function formatStatus(status: string): string {
  switch (status) {
    case "active":
      return "\u2713 Active"
    case "trialing":
      return "\u23f3 Trial"
    case "past_due":
      return "\u26a0 Past Due"
    case "cancelled":
      return "\u2717 Cancelled"
    default:
      return status
  }
}

/**
 * Format USD amount.
 */
function formatUSD(amount: number | null | undefined): string {
  if (amount == null) return "-"
  return `$${amount.toFixed(2)}`
}

/**
 * Format usage details for the modal.
 */
function formatUsageDetails(data: UsageData | null): string {
  if (!data) {
    return "Loading usage data...\n\nPress 'q' to close."
  }

  const lines: string[] = []

  // Plan info section
  lines.push("===== PLAN INFO =====")
  lines.push("")
  lines.push(`Plan:     ${formatPlanName(data.plan_type)}`)
  lines.push(`Status:   ${formatStatus(data.status)}`)
  lines.push("")

  // Features
  lines.push("Features:")
  if (data.limits.unlimited_non_rollout) {
    lines.push("  \u2713 Unlimited non-rollout usage")
  }
  if (data.limits.byok_enabled) {
    lines.push("  \u2713 BYOK enabled (use your own API keys)")
  }
  if (data.limits.team_features_enabled) {
    lines.push("  \u2713 Team collaboration features")
  }
  lines.push("")

  // Rollout credits section (if applicable)
  if (data.plan_type === "pro" || data.plan_type === "team") {
    lines.push("===== ROLLOUT CREDITS =====")
    lines.push("")
    lines.push(`Monthly allowance:  ${formatUSD(data.limits.monthly_rollout_credits_usd)}`)
    lines.push(`Balance:           ${formatUSD(data.rollout_credits_balance_usd)}`)
    lines.push(`Used this period:  ${formatUSD(data.rollout_credits_used_this_period_usd)}`)
    lines.push(`Overdraft limit:   ${formatUSD(data.limits.max_overdraft_usd)}`)
    lines.push("")
  }

  // Usage breakdown (if available)
  if (data.usage_summary) {
    const summary = data.usage_summary
    lines.push("===== USAGE (30 DAYS) =====")
    lines.push("")
    lines.push(`Total cost:    ${formatUSD(summary.total_cost_usd)}`)
    lines.push(`Charged:       ${formatUSD(summary.total_charged_usd)}`)

    // Show savings if on Pro/Team
    if (summary.total_uncharged_usd > 0) {
      lines.push(`Savings:       ${formatUSD(summary.total_uncharged_usd)} \u2714`)
    }
    lines.push("")

    // Breakdown by type
    if (summary.by_type.length > 0) {
      lines.push("By type:")
      for (const item of summary.by_type) {
        const byokNote = item.byok_event_count > 0 ? ` (${item.byok_event_count} BYOK)` : ""
        lines.push(`  ${item.usage_type.padEnd(12)} ${formatUSD(item.total_cost_usd).padStart(10)} (${item.event_count} events${byokNote})`)
      }
    }
  }

  lines.push("")
  lines.push("Press 'q' to close.")

  return lines.join("\n")
}

export function createUsageModal(ctx: AppContext): ModalController & {
  open: () => Promise<void>
  updateContent: () => void
  setData: (data: UsageData | null) => void
} {
  const { ui, renderer } = ctx
  const { appState } = ctx.state

  let usageData: UsageData | null = null

  function toggle(visible: boolean): void {
    ui.usageModalVisible = visible
    ui.usageModalBox.visible = visible
    ui.usageModalTitle.visible = visible
    ui.usageModalText.visible = visible
    ui.usageModalHint.visible = visible
    if (!visible) {
      ui.usageModalText.content = ""
    }
    renderer.requestRender()
  }

  function updateContent(): void {
    if (!ui.usageModalVisible) return

    const raw = formatUsageDetails(usageData)
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.max(20, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, (typeof process.stdout?.rows === "number" ? process.stdout.rows : 40) - 12)

    appState.usageModalOffset = clamp(appState.usageModalOffset || 0, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(appState.usageModalOffset, appState.usageModalOffset + maxLines)

    ui.usageModalTitle.content = `Usage & Plan - ${formatPlanName(usageData?.plan_type || "free")}`
    ui.usageModalText.content = visible.join("\n")
    ui.usageModalHint.content =
      wrapped.length > maxLines
        ? `[${appState.usageModalOffset + 1}-${appState.usageModalOffset + visible.length}/${wrapped.length}] j/k scroll | q close`
        : "j/k scroll | q close"

    renderer.requestRender()
  }

  function setData(data: UsageData | null): void {
    usageData = data
    updateContent()
  }

  async function fetchUsageData(): Promise<void> {
    try {
      // Fetch from the combined usage-plan endpoint
      const response = await apiGetV1("/usage-plan")

      // Map API response to UsageData interface
      const data: UsageData = {
        plan_type: response.plan_type as UsageData["plan_type"],
        status: response.status as UsageData["status"],
        rollout_credits_balance_usd: response.rollout_credits_balance_usd,
        rollout_credits_used_this_period_usd: response.rollout_credits_used_this_period_usd,
        limits: {
          monthly_rollout_credits_usd: response.limits.monthly_rollout_credits_usd,
          max_overdraft_usd: response.limits.max_overdraft_usd,
          unlimited_non_rollout: response.limits.unlimited_non_rollout,
          team_features_enabled: response.limits.team_features_enabled,
          byok_enabled: response.limits.byok_enabled,
        },
        usage_summary: response.usage_summary
          ? {
              total_cost_usd: response.usage_summary.total_cost_usd,
              total_charged_usd: response.usage_summary.total_charged_usd,
              total_uncharged_usd: response.usage_summary.total_uncharged_usd,
              by_type: response.usage_summary.by_type || [],
            }
          : undefined,
      }

      setData(data)
    } catch (err: any) {
      // On error, show free tier as fallback
      const fallbackData: UsageData = {
        plan_type: "free",
        status: "active",
        rollout_credits_balance_usd: null,
        rollout_credits_used_this_period_usd: null,
        limits: {
          monthly_rollout_credits_usd: 0,
          max_overdraft_usd: 0,
          unlimited_non_rollout: false,
          team_features_enabled: false,
          byok_enabled: false,
        },
      }
      setData(fallbackData)

      // Update status to show error
      ctx.state.snapshot.lastError = `Usage fetch failed: ${err?.message || "Unknown error"}`
      ctx.render()
    }
  }

  async function open(): Promise<void> {
    appState.usageModalOffset = 0
    setData(null)  // Show loading state
    toggle(true)
    updateContent()
    await fetchUsageData()
  }

  function handleKey(key: any): boolean {
    if (!ui.usageModalVisible) return false

    if (key.name === "up" || key.name === "k") {
      appState.usageModalOffset = Math.max(0, (appState.usageModalOffset || 0) - 1)
      updateContent()
      return true
    }
    if (key.name === "down" || key.name === "j") {
      appState.usageModalOffset = (appState.usageModalOffset || 0) + 1
      updateContent()
      return true
    }
    if (key.name === "return" || key.name === "enter" || key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true
  }

  return {
    get isVisible() {
      return ui.usageModalVisible
    },
    toggle,
    open,
    updateContent,
    setData,
    handleKey,
  }
}
