import type { SetStoreFunction } from "solid-js/store"

import type { AppContext } from "../context"
import type { AppData } from "../types"
import type { AppState } from "../state/app-state"
import { config, pollingState } from "../state/polling"

export function createSolidContext(
  data: AppData,
  setData: SetStoreFunction<AppData>,
  ui: AppState,
  setUi: SetStoreFunction<AppState>,
): AppContext {
  return {
    state: {
      data,
      ui,
      pollingState,
      config,
    },
    setData,
    setUi,
  }
}
