import { type Accessor, createMemo } from "solid-js"

import { formatActionKeys } from "../../input/keymap"
import { formatPlanName, formatUsageDetails } from "../../formatters/modals"
import type { UsageData } from "./types"
import { ScrollableTextModal } from "./ModalShared"

type UsageModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  usageData: Accessor<UsageData | null>
  offset: number
}

export function UsageModal(props: UsageModalProps) {
  const raw = createMemo(() => formatUsageDetails(props.usageData()))

  return (
    <ScrollableTextModal
      title={`Usage & Plan - ${formatPlanName(props.usageData()?.plan_type || "free")}`.trim()}
      width={72}
      height={28}
      borderColor="#10b981"
      titleColor="#10b981"
      dimensions={props.dimensions}
      raw={raw()}
      offset={props.offset}
      showRangeInTitle="auto"
      hint={{
        scrollHint: `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} scroll`,
        showScrollHint: "always",
        showRange: "never",
        baseHints: [
          `${formatActionKeys("usage.openBilling")} billing`,
          `${formatActionKeys("app.back")} close`,
        ],
      }}
    />
  )
}
