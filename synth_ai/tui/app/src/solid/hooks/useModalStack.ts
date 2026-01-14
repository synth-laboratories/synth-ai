import { createEffect, createMemo, createSignal, type Accessor, type Setter } from "solid-js"

import type { ActiveModal } from "../modals/types"
import { log } from "../../utils/log"

export type ModalStackState = {
  modalStack: Accessor<ActiveModal[]>
  activeModal: Accessor<ActiveModal | null>
  modalInputValue: Accessor<string>
  setModalInputValue: Setter<string>
  setModalInputRef: (ref: any) => void
  openOverlayModal: (kind: ActiveModal) => void
  closeActiveModal: () => void
}

type UseModalStackOptions = {
  abortAction: (key: string) => void
}

export function useModalStack(options: UseModalStackOptions): ModalStackState {
  const [modalStack, setModalStack] = createSignal<ActiveModal[]>([])
  const activeModal = createMemo(() => {
    const stack = modalStack()
    return stack.length ? stack[stack.length - 1] : null
  })
  const [modalInputValue, setModalInputValue] = createSignal("")
  let modalInputRef: any

  const closeActiveModal = (): void => {
    const stack = modalStack()
    const kind = stack[stack.length - 1] ?? null
    log("modal", `close ${kind ?? "none"}`, { stackDepth: stack.length })
    if (kind === "usage") {
      options.abortAction("usage")
    } else if (kind === "sessions") {
      options.abortAction("sessions-refresh")
      options.abortAction("sessions-connect")
      options.abortAction("sessions-disconnect")
    } else if (kind === "task-apps") {
      options.abortAction("task-apps-refresh")
    } else if (kind === "metrics") {
      options.abortAction("metrics")
    }
    try {
      if (modalInputRef && typeof modalInputRef.blur === "function") {
        modalInputRef.blur()
      }
    } catch {
      // Best-effort blur only.
    }
    if (stack.length) {
      setModalStack(stack.slice(0, -1))
    }
    setModalInputValue("")
  }

  const openOverlayModal = (kind: ActiveModal): void => {
    log("modal", `open ${kind}`)
    try {
      if (modalInputRef && typeof modalInputRef.blur === "function") {
        modalInputRef.blur()
      }
    } catch {
      // Best-effort blur only.
    }
    setModalStack((prev) => {
      const next = prev.filter((existing) => existing !== kind)
      next.push(kind)
      log("modal", `change ${kind}`, { stackDepth: next.length })
      return next
    })
  }

  let lastModalKind: ActiveModal | null = null
  createEffect(() => {
    const kind = activeModal()
    if (kind !== lastModalKind) {
      lastModalKind = kind
      if (kind === "filter" || kind === "snapshot") {
        if (modalInputRef) {
          modalInputRef.value = modalInputValue()
          setTimeout(() => modalInputRef.focus(), 1)
        }
      }
    }
  })

  return {
    modalStack,
    activeModal,
    modalInputValue,
    setModalInputValue,
    setModalInputRef: (ref: any) => {
      modalInputRef = ref
    },
    openOverlayModal,
    closeActiveModal,
  }
}
