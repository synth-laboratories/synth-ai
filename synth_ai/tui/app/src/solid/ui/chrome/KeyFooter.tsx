import { For, Show, createMemo, type Accessor } from "solid-js"

import { COLORS } from "../../theme"
import { defaultLayoutSpec } from "../../layout"
import { clampLine } from "../../../utils/text"
import {
  getActionHint,
  buildActionHint,
  buildCombinedHint,
} from "../../../input/keymap"
import { ListPane, type ActivePane, type FocusTarget, type PrimaryView } from "../../../types"

export type JobsLoadMoreState = "none" | "server" | "cache" | "loading"

type KeyFooterProps = {
  primaryView: Accessor<PrimaryView>
  activePane: Accessor<ActivePane>
  focusTarget: Accessor<FocusTarget>
  width: Accessor<number>
  compact: Accessor<boolean>
  jobsLoadMoreState: Accessor<JobsLoadMoreState>
  hasSelectedJob: Accessor<boolean>
  resultsInteractive: Accessor<boolean>
}

type KeyHintItem = {
  hint: string
  active?: boolean
}

// Pre-built combined hints for navigation
const MOVE_HINT = buildCombinedHint("nav.down", "nav.up", "move")
const SCROLL_HINT = buildCombinedHint("nav.down", "nav.up", "scroll")
const FOCUS_HINT = buildCombinedHint("focus.next", "focus.prev", "focus")

export function KeyFooter(props: KeyFooterProps) {
  const hints = createMemo<KeyHintItem[]>(() => {
    const view = props.primaryView()
    const activePane = props.activePane()
    const focusTarget = props.focusTarget()
    const items: KeyHintItem[] = []

    if (focusTarget === "list") {
      items.push({ hint: MOVE_HINT })
      items.push({ hint: buildActionHint("pane.select", "view") })
      if (activePane === ListPane.Jobs) {
        const loadMore = props.jobsLoadMoreState()
        if (loadMore === "server") {
          items.push({ hint: buildActionHint("jobs.loadMore", "more") })
        } else if (loadMore === "cache") {
          items.push({ hint: buildActionHint("jobs.loadMore", "more (cached)") })
        }
        items.push({ hint: getActionHint("modal.open.listFilter") })
      } else if (activePane === ListPane.Logs) {
        items.push({ hint: getActionHint("modal.open.listFilter") })
      }
    } else if (focusTarget === "conversation") {
      if (view === "agent") {
        items.push({ hint: SCROLL_HINT })
      }
    } else if (focusTarget === "principal") {
      if (view === "jobs" || view === "logs") {
        items.push({ hint: SCROLL_HINT })
      }
      if (view === "logs") {
        items.push({ hint: buildActionHint("pane.select", "tail") })
      }
    } else if (focusTarget === "details") {
      items.push({ hint: SCROLL_HINT })
    } else if (focusTarget === "promptDiff") {
      items.push({ hint: SCROLL_HINT })
    } else if (focusTarget === "events") {
      items.push({ hint: MOVE_HINT })
      items.push({ hint: buildActionHint("pane.select", "view") })
      items.push({ hint: getActionHint("modal.open.filter") })
    } else if (focusTarget === "results") {
      if (props.resultsInteractive()) {
        items.push({ hint: MOVE_HINT })
        items.push({ hint: buildActionHint("pane.select", "view") })
      } else {
        items.push({ hint: SCROLL_HINT })
      }
    } else if (focusTarget === "metrics") {
      items.push({ hint: SCROLL_HINT })
      items.push({ hint: buildActionHint("pane.select", "open") })
    } else if (focusTarget === "agent") {
      items.push({ hint: getActionHint("chat.abort") })
      items.push({ hint: getActionHint("app.back") })
    }

    if (view === "jobs" && props.hasSelectedJob()) {
      items.push({ hint: getActionHint("job.cancel") })
      items.push({ hint: buildActionHint("job.artifacts", "download") })
    }

    items.push({ hint: FOCUS_HINT })
    items.push({ hint: getActionHint("app.refresh") })
    items.push({ hint: getActionHint("app.quit") })

    return items
  })

  const hintCount = createMemo(() => hints().length)
  const fullLine = createMemo(() =>
    hints()
      .map((item) => item.hint)
      .join(" | "),
  )
  const compactText = createMemo(() => {
    return clampLine(fullLine(), Math.max(1, props.width() - 2))
  })
  const useCompact = createMemo(() => props.compact() || fullLine().length > props.width() - 2)

  return (
    <box
      height={defaultLayoutSpec.footerHeight}
      backgroundColor={COLORS.bgTabs}
      paddingLeft={1}
      flexDirection="column"
      justifyContent="center"
    >
      <text fg={COLORS.textDim}>Keys:</text>
      <Show
        when={!useCompact()}
        fallback={<text fg={COLORS.textDim}>{compactText()}</text>}
      >
        <box flexDirection="row" gap={1}>
          <For each={hints()}>
            {(item, index) => (
              <>
                <text fg={item.active ? COLORS.textBright : COLORS.textDim}>{item.hint}</text>
                <Show when={index() < hintCount() - 1}>
                  <text fg={COLORS.textDim}>|</text>
                </Show>
              </>
            )}
          </For>
        </box>
      </Show>
    </box>
  )
}
