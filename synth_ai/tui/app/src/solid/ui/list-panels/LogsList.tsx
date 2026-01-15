import { COLORS } from "../../theme"
import { getIndicator } from "../../components/ListCard"
import { ListContainer } from "../../components/ListContainer"
import type { ListWindowItem } from "../../utils/list"
import type { LogsListRow } from "../../hooks/useLogsListState"

interface LogsListProps {
  items: Array<ListWindowItem<LogsListRow>>
  selectedIndex: number
  focused: boolean
  width: number
  height: number
  title: string
  totalCount: number
}

export function LogsList(props: LogsListProps) {
  return (
    <ListContainer
      items={props.items}
      selectedIndex={props.selectedIndex}
      focused={props.focused}
      width={props.width}
      height={props.height}
      title={props.title}
      totalCount={props.totalCount}
      border
      emptyFallback={<text fg={COLORS.textDim}> No log files found.</text>}
      renderItem={(item, ctx) => (
        <box flexDirection="column">
          <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
            <text fg={ctx.fg}>{getIndicator(ctx.isSelected)}{item.type}</text>
          </box>
          <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
            <text fg={ctx.fgDim}>  {item.date}</text>
          </box>
        </box>
      )}
    />
  )
}
