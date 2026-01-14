import { type Accessor } from "solid-js"

import { COLORS } from "../../theme"
import { defaultLayoutSpec } from "../../layout"
import { KeyHint } from "../../components/KeyHint"
import { formatActionKeys } from "../../../input/keymap"
import { ListPane } from "../../../types"
import type { ActivePane, PrincipalPane } from "../../../types"

type AppTabsProps = {
  activePane: Accessor<ActivePane>
  principalPane: Accessor<PrincipalPane>
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
      gap={2}
    >
      <KeyHint description="Create New Job" keyLabel={formatActionKeys("modal.open.createJob", { primaryOnly: true })} />
      <KeyHint description="View Jobs" keyLabel={formatActionKeys("pane.jobs", { primaryOnly: true })} active={props.activePane() === ListPane.Jobs} />
      <KeyHint description="View Logs" keyLabel={formatActionKeys("pane.logs", { primaryOnly: true })} active={props.activePane() === ListPane.Logs} />
      <KeyHint description="Agent" keyLabel={formatActionKeys("pane.togglePrincipal", { primaryOnly: true })} active={props.principalPane() === "opencode"} />
    </box>
  )
}
