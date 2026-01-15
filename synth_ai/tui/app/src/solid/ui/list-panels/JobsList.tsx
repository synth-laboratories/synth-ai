import { Show } from "solid-js"

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
  loadMoreHint: string
}

export function JobsList(props: JobsListProps) {
  const isAtBottom = () => {
    if (!props.items.length || !props.totalCount) return false
    const lastVisible = props.items[props.items.length - 1]
    return lastVisible.globalIndex === props.totalCount - 1
  }

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
      emptyFallback={<text fg={COLORS.textDim}> No jobs yet. Press r to refresh.</text>}
      footer={
        <Show when={props.loadMoreHint && isAtBottom()}>
          <box paddingTop={1}>
            <text fg={COLORS.textDim}>  {props.loadMoreHint}</text>
          </box>
        </Show>
      }
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
