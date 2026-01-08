import type { CliRenderer } from "@opentui/core"
import type { AppContext } from "../context"
import { appState } from "../state/app-state"
import { config, pollingState } from "../state/polling"
import { snapshot } from "../state/snapshot"

export function createSolidContext(onRender: () => void): AppContext {
  return {
    renderer: {} as CliRenderer,
    ui: {} as AppContext["ui"],
    state: {
      snapshot,
      appState,
      pollingState,
      config,
    },
    render: onRender,
    requestRender: onRender,
  }
}
