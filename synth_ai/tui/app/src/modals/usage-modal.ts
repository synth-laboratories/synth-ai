/**
 * Usage modal controller.
 * Shows plan info, rollout credits, and usage breakdown.
 */
import type { AppContext } from "../context"
import { wrapModalText, clamp, type ModalController } from "./base"
import { apiGetV1, getBackendConfig } from "../api/client"
import { openBrowser } from "../auth"
import type { BackendId } from "../types"

export interface UsageData {
  plan_type: "free" | "pro" | "team" | "byok"
  status: "active" | "cancelled" | "past_due" | "trialing" | "inactive"
  access_tier?: string | null  // Access tier: alpha, beta, ga, suspended
  rollout_credits_balance_usd?: number | null
  rollout_credits_used_this_period_usd?: number | null
  byok_providers?: string[]  // List of configured BYOK providers (e.g., ["openai", "anthropic"])
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
    by_job?: Array<{
      job_id: string
      total_cost_usd: number
      charged_cost_usd: number
      uncharged_cost_usd: number
      event_count: number
      byok_event_count: number
      by_type: Array<{
        usage_type: string
        total_cost_usd: number
        charged_cost_usd: number
        uncharged_cost_usd: number
        event_count: number
        byok_event_count: number
      }>
    }>
  }
}

/**
 * Format plan name for display.
 */
function formatPlanName(planType: string): string {
  switch (planType) {
    case "pro":
      return "Pro"
    case "team":
      return "Team"
    case "byok":
      return "BYOK"
    case "free":
    default:
      return "Free"
  }
}

/**
 * Format status for display.
 */
function formatStatus(status: string): string {
  switch (status) {
    case "active":
      return "Active"
    case "trialing":
      return "Trial"
    case "past_due":
      return "Past Due"
    case "cancelled":
      return "Cancelled"
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
  // Add blank lines at the top to prevent overlap with title
  lines.push("")
  lines.push("")

  // Plan info section
  lines.push("===== PLAN INFO =====")
  lines.push("")
  lines.push(`Plan:     ${formatPlanName(data.plan_type)}`)
  lines.push(`Status:   ${formatStatus(data.status)}`)
  
  // Always show access tier (default to "alpha" if not provided)
  const accessTier = data.access_tier || "alpha"
  const tierDisplay = accessTier.charAt(0).toUpperCase() + accessTier.slice(1)
  console.log(`[TUI] Displaying access tier: ${accessTier} -> ${tierDisplay}`)
  lines.push(`Access:   ${tierDisplay}`)
  
  // Always show BYOK status (show "None" if no providers configured)
  if (data.byok_providers && data.byok_providers.length > 0) {
    const providersDisplay = data.byok_providers.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(", ")
    lines.push(`BYOK:     ${providersDisplay}`)
  } else {
    lines.push(`BYOK:     None`)
  }
  
  lines.push("")

  // Features
  lines.push("Features:")
  if (data.limits.unlimited_non_rollout) {
    lines.push("  [*] Unlimited non-rollout usage")
  }
  if (data.limits.byok_enabled) {
    lines.push("  [*] BYOK enabled (use your own API keys)")
  }
  if (data.limits.team_features_enabled) {
    lines.push("  [*] Team collaboration features")
  }
  lines.push("")

  // Rollout credits section (if applicable)
  if (data.plan_type === "pro" || data.plan_type === "team") {
    lines.push("===== ROLLOUT CREDITS =====")
    lines.push("")
    lines.push(`Monthly allowance:  ${formatUSD(data.limits.monthly_rollout_credits_usd)}`)
    lines.push(`Remaining balance:  ${formatUSD(data.rollout_credits_balance_usd)}`)
    lines.push(`Used this period:  ${formatUSD(data.rollout_credits_used_this_period_usd)}`)
    lines.push(`Overdraft limit:   ${formatUSD(data.limits.max_overdraft_usd)}`)
    lines.push("")
  }
  
  // Resource Usage & Billing page instructions (for all plans)
  lines.push("===== RESOURCE USAGE & BILLING =====")
  lines.push("")
  if (data.plan_type === "pro" || data.plan_type === "team") {
    lines.push("Press 'b' to open Usage & Plan page:")
    lines.push("  • View current plan and subscription details")
    lines.push("  • Add rollout credits")
    lines.push("  • Manage payment method")
    lines.push("  • View detailed usage breakdown")
  } else {
    lines.push("Press 'b' to open Usage & Plan page:")
    lines.push("  • View current plan")
    lines.push("  • Upgrade to Pro ($20/month) or Team ($200/month)")
    lines.push("  • Add BYOK keys (use your own API keys)")
    lines.push("  • View detailed usage breakdown")
  }
  lines.push("")

  // Usage breakdown - always show, even if zero
  lines.push("===== USAGE (30 DAYS) =====")
  lines.push("")
  
  if (data.usage_summary) {
    const summary = data.usage_summary
    lines.push(`Total cost:    ${formatUSD(summary.total_cost_usd)}`)
    lines.push(`Charged:       ${formatUSD(summary.total_charged_usd)}`)

    // Show savings if on Pro/Team
    if (summary.total_uncharged_usd > 0) {
      lines.push(`Savings:       ${formatUSD(summary.total_uncharged_usd)}`)
    }
    lines.push("")

    // Breakdown by type
    if (summary.by_type && summary.by_type.length > 0) {
      lines.push("By type:")
      for (const item of summary.by_type) {
        const byokNote = item.byok_event_count > 0 ? ` (${item.byok_event_count} BYOK)` : ""
        lines.push(`  ${item.usage_type.padEnd(12)} ${formatUSD(item.total_cost_usd).padStart(10)} (${item.event_count} events${byokNote})`)
      }
      lines.push("")
    } else {
      lines.push("No usage events in the last 30 days.")
      lines.push("")
    }

    // Breakdown by job
    if (summary.by_job && summary.by_job.length > 0) {
      lines.push("By job:")
      for (const job of summary.by_job) {
        // Truncate job_id for display (show first 20 chars)
        const jobIdDisplay = job.job_id.length > 20 ? job.job_id.substring(0, 20) + "..." : job.job_id
        const byokNote = job.byok_event_count > 0 ? ` (${job.byok_event_count} BYOK)` : ""
        lines.push(`  ${jobIdDisplay.padEnd(24)} ${formatUSD(job.total_cost_usd).padStart(10)} (${job.event_count} events${byokNote})`)
        
        // Show breakdown by type for this job if there are multiple types
        if (job.by_type && job.by_type.length > 1) {
          for (const item of job.by_type) {
            lines.push(`    - ${item.usage_type.padEnd(10)} ${formatUSD(item.total_cost_usd).padStart(10)} (${item.event_count} events)`)
          }
        }
      }
    }
  } else {
    lines.push("Loading usage data...")
    lines.push("(No usage data available)")
  }

  return lines.join("\n")
}

/**
 * Get frontend URL for the current backend.
 */
function getFrontendUrl(backendId: BackendId): string {
  switch (backendId) {
    case "prod":
      return process.env.SYNTH_TUI_FRONTEND_PROD || "https://www.usesynth.ai"
    case "dev":
      return process.env.SYNTH_TUI_FRONTEND_DEV || "https://synth-frontend-dev.onrender.com"
    case "local":
      return process.env.SYNTH_TUI_FRONTEND_LOCAL || "http://localhost:3000"
  }
}

/**
 * Open Usage & Plan page in browser.
 */
function openBillingPage(ctx: AppContext): void {
  try {
    const backendConfig = getBackendConfig(ctx.state.appState.currentBackend)
    const backendId = backendConfig.id
    const frontendUrl = getFrontendUrl(backendId)
    const usageUrl = `${frontendUrl}/usage`
    
    // Open browser
    openBrowser(usageUrl)
    
    // Show status message
    ctx.state.snapshot.status = `✓ Opened Usage & Plan page: ${usageUrl}`
    ctx.render()
    ctx.renderer.requestRender()
  } catch (err: any) {
    ctx.state.snapshot.status = `✗ Failed to open browser: ${err?.message || "Unknown error"}`
    ctx.render()
    ctx.renderer.requestRender()
  }
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
    ui.usageModalHint.visible = false  // Hide separate hint component - hints are in content
    if (visible) {
      ui.jobsSelect.blur()  // Prevent jobs select from handling keys
      updateContent()
    } else {
      ui.usageModalText.content = ""
      if (appState.activePane === "jobs") {
        ui.jobsSelect.focus()  // Restore focus when closing if jobs pane is active
      }
    }
    ctx.render()
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

    // Build hint text and append to content
    const scrollHint = wrapped.length > maxLines
      ? `[${appState.usageModalOffset + 1}-${appState.usageModalOffset + visible.length}/${wrapped.length}] j/k scroll`
      : ""
    
    // Build final hint line - 'b' key works for all plans
    const hintParts: string[] = []
    if (scrollHint) hintParts.push(scrollHint)
    hintParts.push("b usage & plan")  // Always show usage & plan key
    hintParts.push("q close")
    
    const hintLine = hintParts.join(" | ")
    
    // Append hint to visible content
    const contentWithHint = [...visible]
    if (contentWithHint.length > 0) {
      contentWithHint.push("")
      contentWithHint.push(hintLine)
    } else {
      contentWithHint.push(hintLine)
    }

    ui.usageModalTitle.content = `Usage & Plan - ${formatPlanName(usageData?.plan_type || "free")}`
    ui.usageModalText.content = contentWithHint.join("\n")
    
    // Hide the separate hint component (hints are now in content)
    ui.usageModalHint.content = ""
    ui.usageModalHint.visible = false

    ctx.render()
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

      // Debug: log response to see what we're getting
      console.log("[TUI] Usage plan response access_tier:", response.access_tier)

      // Map API response to UsageData interface
      const data: UsageData = {
        plan_type: response.plan_type as UsageData["plan_type"],
        status: response.status as UsageData["status"],
        access_tier: response.access_tier ?? "alpha",  // Default to alpha if not provided
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
        // Backend always returns usage_summary, but ensure we have defaults
        usage_summary: response.usage_summary
          ? {
              total_cost_usd: response.usage_summary.total_cost_usd ?? 0,
              total_charged_usd: response.usage_summary.total_charged_usd ?? 0,
              total_uncharged_usd: response.usage_summary.total_uncharged_usd ?? 0,
              by_type: response.usage_summary.by_type || [],
              by_job: response.usage_summary.by_job || [],
            }
          : {
              // Fallback if backend doesn't return usage_summary
              total_cost_usd: 0,
              total_charged_usd: 0,
              total_uncharged_usd: 0,
              by_type: [],
              by_job: [],
            },
      }

      setData(data)
    } catch (err: any) {
      // On error, show free tier as fallback
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

      // Update status to show error
      ctx.state.snapshot.lastError = `Usage fetch failed: ${err?.message || "Unknown error"}`
      ctx.render()
    }
  }

  async function open(): Promise<void> {
    appState.usageModalOffset = 0
    setData(null)  // Show loading state
    toggle(true)   // Open modal immediately - this must happen first (toggle will blur jobsSelect)
    updateContent()  // Show loading message
    
    // Fetch data in background - errors are handled in fetchUsageData
    await fetchUsageData()
  }

  function handleKey(key: any): boolean {
    if (!ui.usageModalVisible) return false

    // Open billing page (works for all plans) - check this FIRST before other keys
    if (key.name === "b") {
      openBillingPage(ctx)
      // Don't close modal - let user see the status message
      return true
    }

    // Handle scrolling within the modal
    if (key.name === "up" || key.name === "k") {
      appState.usageModalOffset = Math.max(0, (appState.usageModalOffset || 0) - 1)
      updateContent()
      return true  // Consume the key - don't let it affect background
    }
    if (key.name === "down" || key.name === "j") {
      appState.usageModalOffset = (appState.usageModalOffset || 0) + 1
      updateContent()
      return true  // Consume the key - don't let it affect background
    }
    if (key.name === "return" || key.name === "enter" || key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    // Consume all other keys when modal is visible to prevent background interaction
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
