/**
 * Central app context: shared state + render hook.
 *
 * Keeping this in one place prevents circular imports and keeps the Solid
 * app wiring minimal.
 */
import { appState } from "./state/app-state"
import { config, pollingState } from "./state/polling"
import { snapshot } from "./state/snapshot"

export type RenderFn = () => void

export type AppContext = {
  state: {
    snapshot: typeof snapshot
    appState: typeof appState
    pollingState: typeof pollingState
    config: typeof config
  }

  /** Triggers a full UI sync from state. */
  render: RenderFn
}

