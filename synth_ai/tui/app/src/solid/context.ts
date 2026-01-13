import type { AppContext } from "../context"
import { appState } from "../state/app-state"
import { config, pollingState } from "../state/polling"
import { snapshot } from "../state/snapshot"

export function createSolidContext(onRender: () => void): AppContext {
  return {
    state: {
      snapshot,
      appState,
      pollingState,
      config,
    },
    render: onRender,
  }
}
