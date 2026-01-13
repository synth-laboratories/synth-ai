import type {
  SessionHealthResult,
  SessionRecord,
  Snapshot,
  TunnelHealthResult,
  TunnelRecord,
  UsageData,
} from "../types"

export function formatPlanName(planType: string): string {
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

function formatUSD(amount: number | null | undefined): string {
  if (amount == null) return "-"
  return `$${amount.toFixed(2)}`
}

export function formatUsageDetails(data: UsageData | null): string {
  if (!data) {
    return "Loading usage data..."
  }

  const lines: string[] = []
  lines.push("=== PLAN INFO ===")
  lines.push("")
  lines.push(`Plan:     ${formatPlanName(data.plan_type)}`)
  lines.push(`Status:   ${formatStatus(data.status)}`)

  const accessTier = data.access_tier || "alpha"
  lines.push(`Access:   ${accessTier.charAt(0).toUpperCase() + accessTier.slice(1)}`)

  if (data.byok_providers && data.byok_providers.length > 0) {
    const providers = data.byok_providers.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(", ")
    lines.push(`BYOK:     ${providers}`)
  }
  lines.push("")

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

  if (data.plan_type === "pro" || data.plan_type === "team") {
    lines.push("=== ROLLOUT CREDITS ===")
    lines.push("")
    lines.push(`Monthly:   ${formatUSD(data.limits.monthly_rollout_credits_usd)}`)
    lines.push(`Remaining: ${formatUSD(data.rollout_credits_balance_usd)}`)
    lines.push(`Used:      ${formatUSD(data.rollout_credits_used_this_period_usd)}`)
    lines.push("")
  }

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
        lines.push(
          `  ${item.usage_type.padEnd(12)} ${formatUSD(item.total_cost_usd).padStart(10)} (${item.event_count} events${byok})`,
        )
      }
    } else {
      lines.push("No usage in last 30 days.")
    }
  } else {
    lines.push("No usage data available.")
  }

  return lines.join("\n")
}

export function formatTunnelDetails(
  tunnels: TunnelRecord[],
  healthResults: Map<string, TunnelHealthResult>,
  selectedIndex: number,
): string {
  const activeTunnels = tunnels.filter((t) => t.status === "active" && !t.deleted_at)
  if (activeTunnels.length === 0) {
    return "No active task apps (tunnels).\n\nTask apps are Cloudflare managed tunnels that expose\nlocal APIs to the internet for remote execution.\n\nPress 'q' to close."
  }

  const lines: string[] = []
  activeTunnels.forEach((tunnel, idx) => {
    const health = healthResults.get(tunnel.id)
    const isSelected = idx === selectedIndex

    let healthIcon = "?"
    let healthText = "checking..."
    if (health) {
      if (health.healthy) {
        healthIcon = "\u2713"
        healthText = health.response_time_ms != null
          ? `Healthy (${health.response_time_ms}ms)`
          : "Healthy"
      } else {
        healthIcon = "\u2717"
        healthText = health.error?.slice(0, 40) || "Unhealthy"
      }
    }

    const portMatch = tunnel.hostname.match(/task-(\d+)-\d+/)
    const displayPort = portMatch ? portMatch[1] : tunnel.local_port?.toString() || "?"

    const prefix = isSelected ? "> " : "  "
    const hostname = tunnel.hostname.replace(/^https?:\/\//, "")
    const shortHost = hostname.length > 50 ? hostname.slice(0, 47) + "..." : hostname

    lines.push(`${prefix}[${healthIcon}] ${shortHost}`)
    lines.push(`    Port: ${displayPort} | Status: ${healthText}`)
    lines.push(`    Local: ${tunnel.local_host}:${tunnel.local_port}`)
    if (tunnel.created_at) {
      const created = new Date(tunnel.created_at)
      lines.push(`    Created: ${created.toLocaleString()}`)
    }
    if (tunnel.org_name) {
      lines.push(`    Org: ${tunnel.org_name}`)
    }
    lines.push("")
  })

  return lines.join("\n")
}

export function formatSessionDetails(
  sessions: SessionRecord[],
  healthResults: Map<string, SessionHealthResult>,
  selectedIndex: number,
  openCodeUrl: string | null,
): string {
  const activeSessions = sessions.filter(
    (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
  )

  const serverUrl = openCodeUrl || "(not started)"

  if (activeSessions.length === 0) {
    return `No active OpenCode sessions.

Interactive sessions connect to local or remote OpenCode servers
for real-time agent interaction.

OpenCode server: ${serverUrl}

Quick connect:
  Press 'c' to connect to the local OpenCode server
  Press 'C' to connect with custom URL

Press 'q' to close.`
  }

  const lines: string[] = []

  activeSessions.forEach((session, idx) => {
    const health = healthResults.get(session.session_id)
    const isSelected = idx === selectedIndex

    let stateIcon = "?"
    let stateText: string = session.state
    if (session.state === "connected") {
      if (health) {
        if (health.healthy) {
          stateIcon = "\u2713"
          stateText = health.response_time_ms != null
            ? `Connected (${health.response_time_ms}ms)`
            : "Connected"
        } else {
          stateIcon = "\u2717"
          stateText = health.error?.slice(0, 30) || "Unhealthy"
        }
      } else {
        stateIcon = "\u2713"
        stateText = "Connected"
      }
    } else if (session.state === "connecting" || session.state === "reconnecting") {
      stateIcon = "\u21BB"
      stateText = session.state
    } else if (session.state === "error") {
      stateIcon = "\u2717"
      stateText = session.error_message?.slice(0, 30) || "Error"
    }

    const prefix = isSelected ? "> " : "  "
    const localTag = session.is_local ? " [local]" : ""

    lines.push(`${prefix}[${stateIcon}] ${session.session_id}${localTag}`)
    lines.push(`    State: ${stateText}`)
    lines.push(`    Mode: ${session.mode} | Model: ${session.model || "default"}`)

    if (session.opencode_url) {
      const shortUrl = session.opencode_url.length > 50
        ? session.opencode_url.slice(0, 47) + "..."
        : session.opencode_url
      lines.push(`    URL: ${shortUrl}`)
    }
    if (session.tunnel_url && session.tunnel_url !== session.opencode_url) {
      lines.push(`    Tunnel: ${session.tunnel_url}`)
    }

    if (session.connected_at) {
      const connectedAt = new Date(session.connected_at)
      lines.push(`    Connected: ${connectedAt.toLocaleString()}`)
    }
    if (session.last_activity) {
      const lastActivity = new Date(session.last_activity)
      lines.push(`    Last activity: ${lastActivity.toLocaleString()}`)
    }

    lines.push("")
  })

  return lines.join("\n")
}

export function formatConfigMetadata(snapshot: Snapshot): string {
  const job = snapshot.selectedJob
  if (!job) return "(no metadata)"

  const separator = "\u2550\u2550\u2550"
  const lines: string[] = []
  lines.push(`Job: ${job.job_id}`)
  lines.push(`Status: ${job.status}`)
  lines.push(`Type: ${job.training_type || "-"}`)
  lines.push(`Source: ${job.job_source || "unknown"}`)
  lines.push("")

  if (snapshot.lastError && snapshot.status?.includes("Error")) {
    lines.push(`${separator} Error Loading Metadata ${separator}`)
    lines.push(snapshot.lastError)
    lines.push("")
    lines.push("The job details could not be loaded.")
    return lines.join("\n")
  }

  const meta: any = job.metadata
  if (!meta || Object.keys(meta).length === 0) {
    if (snapshot.status?.includes("Loading")) {
      lines.push("Loading job configuration...")
      lines.push("")
      lines.push("Modal will auto-update when loaded.")
    } else if (!job.training_type) {
      lines.push("Loading job configuration...")
      lines.push("")
      lines.push("Press 'i' again after job details finish loading.")
    } else {
      lines.push("No metadata available for this job.")
      lines.push("")
      lines.push(`(job_source: ${job.job_source}, training_type: ${job.training_type})`)
    }
    return lines.join("\n")
  }

  const desc = meta.request_metadata?.description || meta.description
  if (desc) {
    lines.push(`Description: ${desc}`)
    lines.push("")
  }

  const rawConfig =
    meta.prompt_initial_snapshot?.raw_config?.prompt_learning
    || meta.config?.prompt_learning
    || meta.job_config?.prompt_learning
    || meta.prompt_learning
    || meta.config
    || meta.job_config
    || null

  const optimizerConfig = meta.prompt_initial_snapshot?.optimizer_config || meta.optimizer_config || null

  const policy = rawConfig?.policy || optimizerConfig?.policy_config
  if (policy) {
    lines.push(`${separator} Model Configuration ${separator}`)
    if (policy.model) lines.push(`  Model: ${policy.model}`)
    if (policy.provider) lines.push(`  Provider: ${policy.provider}`)
    if (policy.temperature != null) lines.push(`  Temperature: ${policy.temperature}`)
    if (policy.max_completion_tokens) lines.push(`  Max Tokens: ${policy.max_completion_tokens}`)
    lines.push("")
  }

  try {
    const metaJson = JSON.stringify(meta, null, 2)
    if (metaJson.length < 2000) {
      lines.push(`${separator} Raw Metadata ${separator}`)
      lines.push(metaJson)
    }
  } catch {
    // ignore
  }

  return lines.join("\n")
}
