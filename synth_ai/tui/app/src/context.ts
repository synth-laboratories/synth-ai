/**
 * Central app context: shared state + store setters.
 *
 * Keeping this in one place prevents circular imports and keeps the Solid
 * app wiring minimal.
 */
import type { SetStoreFunction } from "solid-js/store"

import type { AppData } from "./types"
import type { AppState } from "./state/app-state"
import { config, pollingState } from "./state/polling"

export type AppContext = {
  state: {
    data: AppData
    ui: AppState
    pollingState: typeof pollingState
    config: typeof config
  }
  setData: SetStoreFunction<AppData>
  setUi: SetStoreFunction<AppState>
}
