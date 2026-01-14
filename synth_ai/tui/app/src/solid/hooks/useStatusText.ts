import { createMemo, type Accessor } from "solid-js"

import type { AppData } from "../../types"
import type { AppState } from "../../state/app-state"

type UseStatusTextOptions = {
  data: AppData
  ui: AppState
}

export function useStatusText(options: UseStatusTextOptions): Accessor<string> {
  return createMemo(() => {
    const status = (options.data.status || "").trim()
    const health = options.ui.healthStatus || "unknown"
    const openCode = options.ui.openCodeStatus
    const base = status ? `${status} | health=${health}` : `health=${health}`
    if (!openCode) return base
    const session = options.ui.openCodeSessionId
    return session
      ? `${base} | opencode=${openCode} (session ${session.slice(-6)})`
      : `${base} | opencode=${openCode}`
  })
}
