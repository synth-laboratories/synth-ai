/**
 * Events list panel component.
 */
import type { EventsPanelProps } from "./types"
import type { JobEvent } from "../../../../tui_data"
import { TEXT } from "../../../theme"
import { getIndicator } from "../../../components/ListCard"
import { ListContainer } from "../../../components/ListContainer"

function padLine(text: string, width: number): string {
  if (text.length >= width) return text
  return text.padEnd(width, " ")
}

function formatEventHeader(event: JobEvent): string {
  const typeRaw = event.type || ""
  return typeRaw.replace(/^prompt\.learning\./, "")
}

function formatEventTimestamp(event: JobEvent): string {
  const ts = (event as any).timestamp
  if (typeof ts !== "string" || ts.length === 0) return ""
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return ""
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}

export function EventsListPanel(props: EventsPanelProps) {
  const lineWidth = () => Math.max(20, props.width - 6)
  const title = () => {
    if (props.totalEvents === 0) return "Events"
    return `Events [${props.selectedIndex + 1}/${props.totalEvents}]`
  }

  return (
    <ListContainer
      items={props.eventItems}
      selectedIndex={props.selectedIndex}
      focused={props.focused}
      title={title()}
      totalCount={props.totalEvents}
      height={props.height}
      flexGrow={props.height ? undefined : 1}
      paddingLeft={1}
      border
      emptyFallback={props.emptyFallback ?? <text fg={TEXT.fg}>No events yet.</text>}
      renderItem={(event, ctx) => {
        const title = formatEventHeader(event)
        const timestamp = formatEventTimestamp(event) || "-"
        const width = lineWidth()
        return (
          <box flexDirection="column">
            <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
              <text fg={ctx.fg}>
                {padLine(`${getIndicator(ctx.isSelected)}${title}`, width)}
              </text>
            </box>
            <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
              <text fg={ctx.fgDim}>
                {padLine(`  ${timestamp}`, width)}
              </text>
            </box>
          </box>
        )
      }}
    />
  )
}
