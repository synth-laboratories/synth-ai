/**
 * Logs detail panel renderer.
 * Renders the right side panel when viewing logs: log file content.
 */
import type { AppContext } from "../../context"
import { renderLogs } from "../logs"

/**
 * Render the logs detail panel (right side).
 */
export function renderLogsDetail(ctx: AppContext): void {
  renderLogs(ctx)
}
