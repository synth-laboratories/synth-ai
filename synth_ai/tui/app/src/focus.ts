import type { KeyAction } from "./input/keymap"
import type { ActivePane, FocusTarget } from "./types"
import { ListPane } from "./types"
type FocusTargetConfig = {
  id: FocusTarget
  order: number
  enabled?: () => boolean
  onFocus?: () => void
  onBlur?: () => void
  handleAction?: (action: KeyAction) => boolean
}

type FocusStateAccess = {
  getFocusTarget: () => FocusTarget
  setFocusTarget: (target: FocusTarget) => void
  getActivePane: () => ActivePane
  setActivePane: (pane: ActivePane) => void
}

let focusState: FocusStateAccess | null = null

function requireFocusState(): FocusStateAccess {
  if (!focusState) {
    throw new Error("Focus state not bound")
  }
  return focusState
}

export function bindFocusState(access: FocusStateAccess): void {
  focusState = access
}

class FocusManager {
  private targets = new Map<FocusTarget, FocusTargetConfig>()
  private order: FocusTarget[] = []

  register(config: FocusTargetConfig): void {
    const exists = this.targets.has(config.id)
    this.targets.set(config.id, config)
    if (!exists) {
      this.order.push(config.id)
    }
    this.order.sort((a, b) => (this.targets.get(a)?.order ?? 0) - (this.targets.get(b)?.order ?? 0))
  }

  current(): FocusTarget {
    return requireFocusState().getFocusTarget()
  }

  isFocused(id: FocusTarget): boolean {
    return requireFocusState().getFocusTarget() === id
  }

  isEnabled(id: FocusTarget): boolean {
    const target = this.targets.get(id)
    if (!target) return false
    return target.enabled ? target.enabled() : true
  }

  setFocus(id: FocusTarget): boolean {
    if (!this.isEnabled(id)) return false
    const state = requireFocusState()
    if (state.getFocusTarget() === id) return false
    const current = this.targets.get(state.getFocusTarget())
    current?.onBlur?.()
    state.setFocusTarget(id)
    this.targets.get(id)?.onFocus?.()
    return true
  }

  ensureValid(): boolean {
    if (this.isEnabled(requireFocusState().getFocusTarget())) return false
    const next = this.firstEnabled()
    if (!next) return false
    return this.setFocus(next)
  }

  focusNext(): boolean {
    return this.shift(1)
  }

  focusPrev(): boolean {
    return this.shift(-1)
  }

  route(action: KeyAction): boolean {
    if (action === "focus.next") return this.focusNext()
    if (action === "focus.prev") return this.focusPrev()
    const current = this.targets.get(requireFocusState().getFocusTarget())
    if (!current?.handleAction) return false
    return current.handleAction(action)
  }

  private firstEnabled(): FocusTarget | null {
    for (const id of this.order) {
      if (this.isEnabled(id)) return id
    }
    return null
  }

  private shift(delta: number): boolean {
    if (!this.order.length) return false
    this.ensureValid()
    const current = requireFocusState().getFocusTarget()
    const startIdx = Math.max(0, this.order.indexOf(current))
    for (let step = 1; step <= this.order.length; step += 1) {
      const idx = (startIdx + delta * step + this.order.length) % this.order.length
      const candidate = this.order[idx]
      if (this.isEnabled(candidate)) {
        return this.setFocus(candidate)
      }
    }
    return false
  }
}

export const focusManager = new FocusManager()

export function setListPane(pane: ActivePane): boolean {
  const state = requireFocusState()
  if (state.getActivePane() === pane) return false
  state.setActivePane(pane)
  if (pane !== ListPane.Jobs && state.getFocusTarget() !== "list") {
    state.setFocusTarget("list")
  }
  focusManager.ensureValid()
  return true
}
