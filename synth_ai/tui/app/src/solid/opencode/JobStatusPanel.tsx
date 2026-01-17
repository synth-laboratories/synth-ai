/**
 * JobStatusPanel - Shows recent job updates in the OpenCode pane
 *
 * Displays a collapsible summary of jobs that changed since the last message.
 */
import { For, Show, createMemo } from "solid-js"
import { COLORS } from "../theme"

export interface JobSummary {
  id: string
  type: string
  previousStatus: string
  currentStatus: string
  updatedAt: Date
}

interface JobStatusPanelProps {
  jobs: JobSummary[]
  visible: boolean
  onDismiss?: () => void
}

function formatRelativeTime(date: Date): string {
  const now = Date.now()
  const diff = now - date.getTime()
  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)

  if (hours > 0) return `${hours}h ago`
  if (minutes > 0) return `${minutes}m ago`
  return `${seconds}s ago`
}

function getStatusColor(status: string): string {
  const s = status.toLowerCase()
  if (s === "completed" || s === "succeeded") return COLORS.success
  if (s === "failed" || s === "error") return COLORS.error
  if (s === "running") return COLORS.textAccent
  return COLORS.textDim
}

/**
 * JobStatusPanel displays recent job updates in a compact format.
 */
export function JobStatusPanel(props: JobStatusPanelProps) {
  const displayJobs = createMemo(() => props.jobs.slice(0, 3))

  return (
    <Show when={props.visible && props.jobs.length > 0}>
      <box
        flexDirection="column"
        borderStyle="rounded"
        borderColor={COLORS.border}
        paddingLeft={1}
        paddingRight={1}
        marginBottom={1}
      >
        <box flexDirection="row" justifyContent="space-between">
          <text fg={COLORS.text}>
            <span style={{ bold: true }}>Recent Job Updates</span>
          </text>
          <Show when={props.onDismiss}>
            <text fg={COLORS.textDim}>(auto-hide in 10s)</text>
          </Show>
        </box>

        <For each={displayJobs()}>
          {(job) => (
            <box flexDirection="column" marginTop={0}>
              <text fg={COLORS.text}>
                <span style={{ fg: COLORS.textDim }}>{"• "}</span>
                {job.id.slice(0, 20)}{job.id.length > 20 ? "..." : ""}
              </text>
              <text fg={COLORS.textDim}>
                {"  Status: "}
                <span style={{ fg: getStatusColor(job.previousStatus) }}>{job.previousStatus}</span>
                {" → "}
                <span style={{ fg: getStatusColor(job.currentStatus) }}>{job.currentStatus}</span>
              </text>
              <text fg={COLORS.textDim}>
                {"  Updated: "}{formatRelativeTime(job.updatedAt)}
              </text>
            </box>
          )}
        </For>
      </box>
    </Show>
  )
}
