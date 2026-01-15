import { type Accessor, createMemo } from "solid-js"

import { formatActionKeys } from "../../input/keymap"
import { formatConfigMetadata } from "../../formatters/modals"
import type { AppData } from "../../types"
import { ScrollableTextModal } from "./ModalShared"

type ConfigModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  data: AppData
  offset: number
}

export function ConfigModal(props: ConfigModalProps) {
  const raw = createMemo(() => formatConfigMetadata(props.data))

  return (
    <ScrollableTextModal
      title="Job Configuration"
      width={100}
      height={24}
      borderColor="#f59e0b"
      titleColor="#f59e0b"
      dimensions={props.dimensions}
      raw={raw()}
      offset={props.offset}
      hint={{
        scrollHint: `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} scroll`,
        baseHints: [`${formatActionKeys("app.back")} close`],
      }}
    />
  )
}
