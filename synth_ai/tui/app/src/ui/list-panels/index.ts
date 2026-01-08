/**
 * List panel configurations registry.
 */
import type { AppContext } from "../../context"
import type { ListPanelConfig, ListPanelId } from "../../components/list-panel"
import { createJobsListConfig } from "./jobs-list"
import { createLogsListConfig } from "./logs-list"

export { createJobsListConfig } from "./jobs-list"
export { createLogsListConfig } from "./logs-list"

/**
 * Get the list panel configuration for the given panel ID.
 */
export function getListPanelConfig(
  ctx: AppContext,
  panelId: ListPanelId
): ListPanelConfig<any> {
  switch (panelId) {
    case "jobs":
      return createJobsListConfig(ctx)
    case "logs":
      return createLogsListConfig(ctx)
  }
}
