import { type Accessor, createMemo } from "solid-js"

import { formatActionKeys } from "../../input/keymap"
import { formatTunnelDetails } from "../../formatters/modals"
import type { AppData } from "../../types"
import { ScrollableTextModal } from "./ModalShared"

type TaskAppsModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  data: AppData
  selectedIndex: number
  offset: number
}

export function TaskAppsModal(props: TaskAppsModalProps) {
  const raw = createMemo(() =>
    formatTunnelDetails(props.data.tunnels, props.data.tunnelHealthResults, props.selectedIndex),
  )
  const title = createMemo(() => {
    const total = props.data.tunnels.length
    return `Task Apps (${total} tunnel${total !== 1 ? "s" : ""})`
  })

  return (
    <ScrollableTextModal
      title={title()}
      width={90}
      height={20}
      borderColor="#06b6d4"
      titleColor="#06b6d4"
      dimensions={props.dimensions}
      raw={raw()}
      offset={props.offset}
      hint={{
        baseHints: [
          `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} select`,
          `${formatActionKeys("modal.copy", { primaryOnly: true })} copy hostname`,
          `${formatActionKeys("app.back")} close`,
        ],
      }}
    />
  )
}
