import { createEffect, createMemo, createSignal, onCleanup, type Accessor, type Setter } from "solid-js"
import path from "node:path"

import type { LayoutMetrics } from "../layout"
import type { JobEvent } from "../../tui_data"
import type { ModalState } from "../modals/types"
import { formatEventDetail } from "../../formatters"
import { registerCleanup, unregisterCleanup } from "../../lifecycle"
import { clamp, wrapModalText } from "../../utils/truncate"
import { formatActionKeys } from "../../input/keymap"
import { readLogFile } from "../utils/logs"

export type DetailModalLayout = {
  width: number
  height: number
  left: number
  top: number
}

export type DetailModalView = {
  total: number
  offset: number
  maxOffset: number
  visible: string[]
  visibleCount: number
}

export type DetailModalState = {
  modal: Accessor<ModalState | null>
  setModal: Setter<ModalState | null>
  modalLayout: Accessor<DetailModalLayout>
  modalView: Accessor<DetailModalView | null>
  modalHint: Accessor<string>
  openEventModal: (event: JobEvent) => void
  openLogModal: (filePath: string) => Promise<void>
}

type UseDetailModalOptions = {
  layout: Accessor<LayoutMetrics>
}

export function useDetailModal(options: UseDetailModalOptions): DetailModalState {
  const [modal, setModal] = createSignal<ModalState | null>(null)

  const modalLayout = createMemo((): DetailModalLayout => {
    const state = modal()
    if (state?.fullscreen) {
      return {
        width: Math.max(1, options.layout().totalWidth),
        height: Math.max(1, options.layout().totalHeight),
        left: 0,
        top: 0,
      }
    }
    const width = Math.min(100, Math.max(40, options.layout().totalWidth - 4))
    const height = Math.min(26, Math.max(12, options.layout().totalHeight - 6))
    const left = Math.max(0, Math.floor((options.layout().totalWidth - width) / 2))
    const top = Math.max(1, Math.floor((options.layout().totalHeight - height) / 2))
    return { width, height, left, top }
  })
  const modalBodyHeight = createMemo(() => Math.max(1, modalLayout().height - 4))
  const modalLines = createMemo(() => {
    const state = modal()
    if (!state) return []
    const maxWidth = Math.max(10, modalLayout().width - 4)
    return wrapModalText(state.raw, maxWidth)
  })
  const modalView = createMemo((): DetailModalView | null => {
    const state = modal()
    if (!state) return null
    const lines = modalLines()
    const maxOffset = Math.max(0, lines.length - modalBodyHeight())
    const offset = clamp(state.offset, 0, maxOffset)
    const resolvedOffset = state.type === "log" && state.tail ? maxOffset : offset
    const visible = lines.slice(resolvedOffset, resolvedOffset + modalBodyHeight())
    return {
      total: lines.length,
      offset: resolvedOffset,
      maxOffset,
      visible,
      visibleCount: modalBodyHeight(),
    }
  })
  const modalHint = createMemo(() => {
    const state = modal()
    const view = modalView()
    if (!state || !view) return ""
    const range = view.total > view.visibleCount
      ? `[${view.offset + 1}-${Math.min(view.offset + view.visible.length, view.total)}/${view.total}] `
      : ""
    const fullscreenHint = `${formatActionKeys("detail.toggleFullscreen", { primaryOnly: true })} fullscreen | `
    const scrollHint = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} scroll`
    if (state.type === "log") {
      const tail = state.tail ? " [TAIL]" : ""
      return `${range}${fullscreenHint}${scrollHint} | ${formatActionKeys("detail.tail", { primaryOnly: true })} tail${tail} | ${formatActionKeys("modal.copy", { primaryOnly: true })} copy | ${formatActionKeys("app.back")} close`
    }
    return `${range}${fullscreenHint}${scrollHint} | ${formatActionKeys("app.back")} close`
  })

  createEffect(() => {
    const current = modal()
    if (!current || current.type !== "log") return
    const filePath = current.path
    let disposed = false
    const timer = setInterval(() => {
      void readLogFile(filePath).then((raw) => {
        if (disposed) return
        setModal((prev) => {
          if (!prev || prev.type !== "log") return prev
          return { ...prev, raw }
        })
      })
    }, 1000)
    const cleanupName = "log-modal-refresh-interval"
    const cleanup = () => {
      clearInterval(timer)
    }
    registerCleanup(cleanupName, cleanup)
    onCleanup(() => {
      disposed = true
      cleanup()
      unregisterCleanup(cleanupName)
    })
  })

  createEffect(() => {
    const state = modal()
    if (!state) return
    const lines = modalLines()
    const maxOffset = Math.max(0, lines.length - modalBodyHeight())
    let nextOffset = state.offset
    if (state.type === "log" && state.tail) {
      nextOffset = maxOffset
    }
    nextOffset = clamp(nextOffset, 0, maxOffset)
    if (nextOffset !== state.offset) {
      setModal({ ...state, offset: nextOffset })
    }
  })

  const openEventModal = (event: JobEvent): void => {
    const detail = event.message ?? formatEventDetail(event.data)
    const header = `${event.type} (seq ${event.seq})`
    const raw = detail ? `${header}\n\n${detail}` : header
    setModal({
      type: "event",
      title: "Event Detail",
      raw,
      offset: 0,
    })
  }

  const openLogModal = async (filePath: string): Promise<void> => {
    const raw = await readLogFile(filePath)
    setModal({
      type: "log",
      title: `Log: ${path.basename(filePath)}`,
      raw,
      offset: 0,
      tail: true,
      path: filePath,
    })
  }

  return {
    modal,
    setModal,
    modalLayout,
    modalView,
    modalHint,
    openEventModal,
    openLogModal,
  }
}
