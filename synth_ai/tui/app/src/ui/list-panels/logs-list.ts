/**
 * Log files list panel configuration.
 */
import type { AppContext } from "../../context"
import type { ListPanelConfig, ListPanelItem } from "../../components/list-panel"
import { listLogFiles, type LogFileInfo } from "../logs"

function getLogType(name: string): string {
  if (name.includes("_deploy_")) return "deploy"
  if (name.includes("_serve_")) return "serve"
  return "log"
}

function formatLogDate(name: string): string {
  const match = name.match(/(\d{4}_\d{2}_\d{2})_([0-9]{2}[:\-][0-9]{2}[:\-][0-9]{2})/)
  if (match) {
    const date = match[1].replace(/_/g, "-")
    const time = match[2].replace(/-/g, ":")
    return `${date} ${time}`
  }
  return name
}

export function createLogsListConfig(ctx: AppContext): ListPanelConfig<LogFileInfo> {
  return {
    id: "logs",
    title: "Logs",
    emptyMessage: "No log files found",
    formatItem: (file: LogFileInfo): ListPanelItem => ({
      id: file.path,
      name: getLogType(file.name),
      description: formatLogDate(file.name),
    }),
    getItems: () => listLogFiles(),
    onSelect: (file: LogFileInfo) => {
      // Store the selected log file path for the detail view
      const files = listLogFiles()
      const newIndex = files.findIndex((f) => f.path === file.path)
      if (newIndex !== ctx.state.appState.logsSelectedIndex) {
        ctx.state.appState.logsSelectedIndex = newIndex
        // Reset scroll position when selecting a new file
        ctx.state.appState.logsContentOffset = 0
        ctx.state.appState.logsContentTailMode = true
      }
    },
  }
}
