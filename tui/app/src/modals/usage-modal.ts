/**
 * Usage modal controller.
 * Shows plan info, rollout credits, and usage breakdown.
 * Adapted for nightly's focusManager and createModalUI patterns.
 */
import type { AppContext } from "../context"
import { createModalUI, wrapModalText, clamp, type ModalController, type ModalUI } from "./base"
import { focusManager } from "../focus"
import { apiGetV1 } from "../api/client"
import { openBrowser } from "../auth"
import { appState, getFrontendUrl } from "../state/app-state"
import { getAbortSignal } from "../lifecycle/shutdown"

export interface UsageData {
  plan_type: "free" | "pro" | "team"
  status: "active" | "cancelled" | "past_due" | "trialing" | "inactive"
  access_tier?: string | null
  rollout_credits_balance_usd?: number | null
  rollout_credits_used_this_period_usd?: number | null
  byok_providers?: string[]
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
    total_uncharged_usd: number
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

function formatPlanName(planType: string): string {
  switch (planType) {
    case "pro": return "Pro"
    case "team": return "Team"
    case "free":
    default: return "Free"
  }
}

function formatStatus(status: string): string {
  switch (status) {
    case "active": return "Active"
    case "trialing": return "Trial"
    case "past_due": return "Past Due"
    case "cancelled": return "Cancelled"
    default: return status
  }
}

function formatUSD(amount: number | null | undefined): string {
  if (amount == null) return "-"
  return `$${amount.toFixed(2)}`
}

function formatUsageDetails(data: UsageData | null): string {
  if (!data) {
    return "Loading usage data..."
  }

  const lines: string[] = []

  // Plan info section
  lines.push("=== PLAN INFO ===")
  lines.push("")
  lines.push(`Plan:     ${formatPlanName(data.plan_type)}`)
  lines.push(`Status:   ${formatStatus(data.status)}`)

  const accessTier = data.access_tier || "alpha"
  lines.push(`Access:   ${accessTier.charAt(0).toUpperCase() + accessTier.slice(1)}`)

  if (data.byok_providers && data.byok_providers.length > 0) {
    const providers = data.byok_providers.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(", ")
    lines.push(`BYOK:     ${providers}`)
  }
  lines.push("")

  // Features
  lines.push("Features:")
  if (data.limits.unlimited_non_rollout) {
    lines.push("  [*] Unlimited non-rollout usage")
  }
  if (data.limits.byok_enabled) {
    lines.push("  [*] BYOK enabled")
  }
  if (data.limits.team_features_enabled) {
    lines.push("  [*] Team features")
  }
  lines.push("")

  // Rollout credits (if applicable)
  if (data.plan_type === "pro" || data.plan_type === "team") {
    lines.push("=== ROLLOUT CREDITS ===")
    lines.push("")
    lines.push(`Monthly:   ${formatUSD(data.limits.monthly_rollout_credits_usd)}`)
    lines.push(`Remaining: ${formatUSD(data.rollout_credits_balance_usd)}`)
    lines.push(`Used:      ${formatUSD(data.rollout_credits_used_this_period_usd)}`)
    lines.push("")
  }

  // Usage breakdown
  lines.push("=== USAGE (30 DAYS) ===")
  lines.push("")

  if (data.usage_summary) {
    const summary = data.usage_summary
    lines.push(`Total:   ${formatUSD(summary.total_cost_usd)}`)
    lines.push(`Charged: ${formatUSD(summary.total_charged_usd)}`)
    if (summary.total_uncharged_usd > 0) {
      lines.push(`Savings: ${formatUSD(summary.total_uncharged_usd)}`)
    }
    lines.push("")

    if (summary.by_type && summary.by_type.length > 0) {
      lines.push("By type:")
      for (const item of summary.by_type) {
        const byok = item.byok_event_count > 0 ? ` (${item.byok_event_count} BYOK)` : ""
        lines.push(`  ${item.usage_type.padEnd(12)} ${formatUSD(item.total_cost_usd).padStart(10)} (${item.event_count} events${byok})`)
      }
    } else {
      lines.push("No usage in last 30 days.")
    }
  } else {
    lines.push("No usage data available.")
  }

  return lines.join("\n")
}

export type UsageModalController = ModalController & {
  open: () => Promise<void>
  setData: (data: UsageData | null) => void
}

export function createUsageModal(ctx: AppContext): UsageModalController {
  const { renderer } = ctx

  let usageData: UsageData | null = null

  // Create modal UI using the primitive
  const modal: ModalUI = createModalUI(renderer, {
    id: "usage-modal",
    width: 72,
    height: 28,
    borderColor: "#10b981",
    titleColor: "#10b981",
    zIndex: 10,
  })

  modal.setTitle("Usage & Plan")
  modal.setHint("j/k scroll  b open billing  q close")

  function updateContent(): void {
    const raw = formatUsageDetails(usageData)
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const maxWidth = Math.min(68, cols - 20)
    const wrapped = wrapModalText(raw, maxWidth)
    const maxLines = Math.max(1, 22) // Fixed height for consistency

    appState.usageModalOffset = clamp(appState.usageModalOffset || 0, 0, Math.max(0, wrapped.length - maxLines))
    const visible = wrapped.slice(appState.usageModalOffset, appState.usageModalOffset + maxLines)

    const scrollIndicator = wrapped.length > maxLines
      ? `[${appState.usageModalOffset + 1}-${appState.usageModalOffset + visible.length}/${wrapped.length}]`
      : ""

    modal.setTitle(`Usage & Plan - ${formatPlanName(usageData?.plan_type || "free")} ${scrollIndicator}`)
    modal.setContent(visible.join("\n"))
    renderer.requestRender()
  }

  function setData(data: UsageData | null): void {
    usageData = data
    updateContent()
  }

  async function fetchUsageData(): Promise<void> {
    try {
      const response = await apiGetV1("/usage-plan", { signal: getAbortSignal() })

      const data: UsageData = {
        plan_type: response.plan_type as UsageData["plan_type"],
        status: response.status as UsageData["status"],
        access_tier: response.access_tier ?? "alpha",
        rollout_credits_balance_usd: response.rollout_credits_balance_usd ?? null,
        rollout_credits_used_this_period_usd: response.rollout_credits_used_this_period_usd ?? null,
        byok_providers: response.byok_providers || [],
        limits: {
          monthly_rollout_credits_usd: response.limits?.monthly_rollout_credits_usd ?? 0,
          max_overdraft_usd: response.limits?.max_overdraft_usd ?? 0,
          unlimited_non_rollout: response.limits?.unlimited_non_rollout ?? false,
          team_features_enabled: response.limits?.team_features_enabled ?? false,
          byok_enabled: response.limits?.byok_enabled ?? false,
        },
        usage_summary: response.usage_summary
          ? {
              total_cost_usd: response.usage_summary.total_cost_usd ?? 0,
              total_charged_usd: response.usage_summary.total_charged_usd ?? 0,
              total_uncharged_usd: response.usage_summary.total_uncharged_usd ?? 0,
              by_type: response.usage_summary.by_type || [],
            }
          : undefined,
      }

      setData(data)
    } catch (err: any) {
      // Fallback to free plan on error
      const fallbackData: UsageData = {
        plan_type: "free",
        status: "active",
        rollout_credits_balance_usd: null,
        rollout_credits_used_this_period_usd: null,
        byok_providers: [],
        limits: {
          monthly_rollout_credits_usd: 0,
          max_overdraft_usd: 0,
          unlimited_non_rollout: false,
          team_features_enabled: false,
          byok_enabled: false,
        },
      }
      setData(fallbackData)
      ctx.state.snapshot.lastError = `Usage fetch failed: ${err?.message || "Unknown"}`
      ctx.render()
    }
  }

  function openBillingPage(): void {
    try {
      const frontendUrl = getFrontendUrl(appState.currentBackend)
      const usageUrl = `${frontendUrl}/usage`
      openBrowser(usageUrl)
      ctx.state.snapshot.status = `Opened: ${usageUrl}`
      ctx.render()
    } catch (err: any) {
      ctx.state.snapshot.status = `Failed to open browser: ${err?.message || "Unknown"}`
      ctx.render()
    }
  }

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "usage-modal",
        handleKey,
      })
      modal.center()
    } else {
      focusManager.pop("usage-modal")
    }
    modal.setVisible(visible)
  }

  async function open(): Promise<void> {
    appState.usageModalOffset = 0
    setData(null) // Show loading state
    toggle(true)
    updateContent()
    await fetchUsageData()
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false

    if (key.name === "b") {
      openBillingPage()
      return true
    }
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
    return true // Consume all keys when modal is open
  }

  return {
    get isVisible() {
      return modal.visible
    },
    toggle,
    open,
    setData,
    handleKey,
  }
}
