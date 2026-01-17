import { type Accessor, createMemo } from "solid-js"

import { getActionHint, buildCombinedHint } from "../../input/keymap"
import { formatMetricsCharts } from "../../formatters/metrics"
import type { AppData } from "../../types"
import { ScrollableTextModal } from "./ModalShared"

type MetricsModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  data: AppData
  offset: number
}

export function MetricsModal(props: MetricsModalProps) {
  const metrics = createMemo(() => props.data.metrics || null)
  const pointCount = createMemo(() => {
    const m = metrics()
    if (!m) return 0
    const points = Array.isArray(m.points) ? m.points : []
    return points.length
  })
  const modalWidth = createMemo(() => props.dimensions().width - 4)
  const modalHeight = createMemo(() => props.dimensions().height - 6)
  const raw = createMemo(() =>
    formatMetricsCharts(props.data.metrics, {
      width: props.dimensions().width - 6,
      height: props.dimensions().height - 8,
    }),
  )

  return (
    <ScrollableTextModal
      title={`Metrics (${pointCount()} points)`}
      width={modalWidth()}
      height={modalHeight()}
      borderColor="#8b5cf6"
      titleColor="#8b5cf6"
      dimensions={props.dimensions}
      raw={raw()}
      offset={props.offset}
      hint={{
        scrollHint: buildCombinedHint("nav.down", "nav.up", "scroll"),
        baseHints: [
          getActionHint("metrics.refresh"),
          getActionHint("app.back"),
        ],
      }}
    />
  )
}
