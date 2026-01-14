import { For, Show } from "solid-js"
import { COLORS } from "../../theme"
import { ListCard, getIndicator } from "../../components/ListCard"
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
  const items = () => props.items
  return (
    <box
      width={props.width}
      height={props.height}
      borderStyle="single"
      borderColor={props.focused ? COLORS.textAccent : COLORS.border}
      title={props.title}
      titleAlignment="left"
      flexDirection="column"
    >
      <Show
        when={props.totalCount > 0}
        fallback={<text fg={COLORS.textDim}> No jobs yet. Press r to refresh.</text>}
      >
        <For each={items()}>
          {(entry) => {
            const isSelected = entry.globalIndex === props.selectedIndex
            const item = entry.item

            return (
              <ListCard isSelected={isSelected}>
                {(ctx) => (
                  <box flexDirection="column">
                    <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
                      <text fg={ctx.fg}>{getIndicator(ctx.isSelected)}{item.type}</text>
                    </box>
                    <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
                      <text fg={ctx.fgDim}>  {item.status} | {item.date}</text>
                    </box>
                  </box>
                )}
              </ListCard>
            )
          }}
        </For>
      </Show>
    </box>
  )
}
