import { For, Show, createMemo, type Accessor } from "solid-js"

import { COLORS } from "../../theme"
import { defaultLayoutSpec } from "../../layout"
import { KeyHint } from "../../components/KeyHint"
import { formatActionKeys } from "../../../input/keymap"
import { ListPane, type ActivePane, type FocusTarget, type PrincipalPane } from "../../../types"

export type JobsLoadMoreState = "none" | "server" | "cache" | "loading"

type KeyFooterProps = {
  principalPane: Accessor<PrincipalPane>
  activePane: Accessor<ActivePane>
  focusTarget: Accessor<FocusTarget>
  jobsLoadMoreState: Accessor<JobsLoadMoreState>
  hasSelectedJob: Accessor<boolean>
}

type KeyHintItem = {
  description: string
  keyLabel: string
  active?: boolean
}

const NAV_KEYS = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })}`
const FOCUS_KEYS = `${formatActionKeys("focus.next", { primaryOnly: true })}/${formatActionKeys("focus.prev", { primaryOnly: true })}`
const SELECT_KEY = formatActionKeys("pane.select", { primaryOnly: true })
const REFRESH_KEY = formatActionKeys("app.refresh", { primaryOnly: true })
const QUIT_KEY = formatActionKeys("app.quit", { primaryOnly: true })
const BACK_KEY = formatActionKeys("app.back", { primaryOnly: true })
const SESSIONS_KEY = formatActionKeys("pane.openSessions", { primaryOnly: true })
const LOAD_MORE_KEY = formatActionKeys("jobs.loadMore", { primaryOnly: true })
const LIST_FILTER_KEY = formatActionKeys("modal.open.listFilter", { primaryOnly: true })
const EVENT_FILTER_KEY = formatActionKeys("modal.open.filter", { primaryOnly: true })
const CANCEL_KEY = formatActionKeys("job.cancel", { primaryOnly: true })
const ARTIFACTS_KEY = formatActionKeys("job.artifacts", { primaryOnly: true })

export function KeyFooter(props: KeyFooterProps) {
  const hints = createMemo<KeyHintItem[]>(() => {
    if (props.principalPane() === "opencode") {
      return [
        { description: "back", keyLabel: BACK_KEY },
        { description: "sessions", keyLabel: SESSIONS_KEY },
        { description: "quit", keyLabel: QUIT_KEY },
      ]
    }

    const activePane = props.activePane()
    const focusTarget = props.focusTarget()
    const items: KeyHintItem[] = []

    if (focusTarget === "list") {
      items.push({ description: "move", keyLabel: NAV_KEYS })
      items.push({ description: "view", keyLabel: SELECT_KEY })
      if (activePane === ListPane.Jobs) {
        const loadMore = props.jobsLoadMoreState()
        if (loadMore === "server") {
          items.push({ description: "more", keyLabel: LOAD_MORE_KEY })
        } else if (loadMore === "cache") {
          items.push({ description: "more (cached)", keyLabel: LOAD_MORE_KEY })
        }
        items.push({ description: "filter", keyLabel: LIST_FILTER_KEY })
      }
    } else if (focusTarget === "events") {
      items.push({ description: "move", keyLabel: NAV_KEYS })
      items.push({ description: "view", keyLabel: SELECT_KEY })
      items.push({ description: "filter", keyLabel: EVENT_FILTER_KEY })
    } else if (focusTarget === "results") {
      items.push({ description: "move", keyLabel: NAV_KEYS })
      items.push({ description: "view", keyLabel: SELECT_KEY })
    } else if (focusTarget === "metrics") {
      items.push({ description: "open", keyLabel: SELECT_KEY })
    } else if (focusTarget === "logs-detail") {
      items.push({ description: "scroll", keyLabel: NAV_KEYS })
      items.push({ description: "tail", keyLabel: SELECT_KEY })
    }

    if (activePane === ListPane.Jobs && props.hasSelectedJob()) {
      items.push({ description: "cancel", keyLabel: CANCEL_KEY })
      items.push({ description: "artifacts", keyLabel: ARTIFACTS_KEY })
    }

    items.push({ description: "focus", keyLabel: FOCUS_KEYS })
    items.push({ description: "refresh", keyLabel: REFRESH_KEY })
    items.push({ description: "quit", keyLabel: QUIT_KEY })

    return items
  })

  const hintCount = createMemo(() => hints().length)

  return (
    <box
      height={defaultLayoutSpec.footerHeight}
      backgroundColor={COLORS.bgTabs}
      paddingLeft={1}
      alignItems="center"
      flexDirection="row"
      gap={2}
    >
      <text fg={COLORS.textDim}>Keys: </text>
      <box flexDirection="row" gap={1}>
        <For each={hints()}>
          {(item, index) => (
            <>
              <KeyHint description={item.description} keyLabel={item.keyLabel} active={item.active} />
              <Show when={index() < hintCount() - 1}>
                <text fg={COLORS.textDim}>|</text>
              </Show>
            </>
          )}
        </For>
      </box>
    </box>
  )
}
