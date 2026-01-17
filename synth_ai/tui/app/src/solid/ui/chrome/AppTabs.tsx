import { Show, type Accessor } from "solid-js"

import { COLORS } from "../../theme"
import { defaultLayoutSpec } from "../../layout"
import { KeyHint } from "../../components/KeyHint"
import type { PrimaryView } from "../../../types"

type AppTabsProps = {
  primaryView: Accessor<PrimaryView>
  compact: Accessor<boolean>
}

export function AppTabs(props: AppTabsProps) {
  return (
    <box
      height={defaultLayoutSpec.tabsHeight}
      backgroundColor={COLORS.bgTabs}
      border
      borderStyle="single"
      borderColor={COLORS.borderDim}
      alignItems="center"
      flexDirection="row"
      gap={props.compact() ? 1 : 3}
    >
      <Show when={!props.compact()}>
        <KeyHint action="modal.open.createJob" />
      </Show>
      <KeyHint action="pane.jobs" active={props.primaryView() === "jobs"} />
      <KeyHint action="pane.agent" active={props.primaryView() === "agent"} />
      <KeyHint action="pane.logs" active={props.primaryView() === "logs"} />
    </box>
  )
}
