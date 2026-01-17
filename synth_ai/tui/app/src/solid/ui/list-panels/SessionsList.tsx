import { COLORS } from "../../theme"
import { getIndicator } from "../../components/ListCard"
import { ListContainer } from "../../components/ListContainer"
import type { ListWindowItem } from "../../utils/list"
import type { SessionsListRow } from "../../hooks/useSessionsListState"
import { clampLine } from "../../../utils/text"

interface SessionsListProps {
  items: Array<ListWindowItem<SessionsListRow>>
  selectedIndex: number
  focused: boolean
  width: number
  height: number
  title: string
  totalCount: number
}

export function SessionsList(props: SessionsListProps) {
  const lineWidth = () => Math.max(10, props.width - 4)

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
      emptyFallback={<text fg={COLORS.textDim}> No sessions yet.</text>}
      renderItem={(item, ctx) => (
        <box flexDirection="column">
          <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
            <text fg={ctx.fg}>{clampLine(`${getIndicator(ctx.isSelected)}${item.title}`, lineWidth())}</text>
          </box>
          <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
            <text fg={ctx.fgDim}>{clampLine(`  ${item.detail}`, lineWidth())}</text>
          </box>
        </box>
      )}
    />
  )
}
