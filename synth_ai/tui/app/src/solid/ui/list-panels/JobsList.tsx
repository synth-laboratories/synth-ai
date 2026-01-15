import { COLORS } from "../../theme"
import { getIndicator } from "../../components/ListCard"
import { ListContainer } from "../../components/ListContainer"
import type { ListWindowItem } from "../../utils/list"
import type { JobsListRow } from "../../hooks/useJobsListState"

interface JobsListProps {
  items: Array<ListWindowItem<JobsListRow>>
  selectedIndex: number
  focused: boolean
  width: number
  height: number
  title: string
  totalCount: number
}

export function JobsList(props: JobsListProps) {
  return (
    <ListContainer
      items={props.items}
      selectedIndex={props.selectedIndex}
      focused={props.focused}
      width={props.width}
      height={props.height}
      title={props.title}
      totalCount={props.totalCount}
      emptyFallback={<text fg={COLORS.textDim}> No jobs yet. Press r to refresh.</text>}
      renderItem={(item, ctx) => (
        <box flexDirection="column">
          <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
            <text fg={ctx.fg}>{getIndicator(ctx.isSelected)}{item.type}</text>
          </box>
          <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
            <text fg={ctx.fgDim}>  {item.status} | {item.date}</text>
          </box>
        </box>
      )}
    />
  )
}
