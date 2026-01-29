/**
 * Central app context: a single object that wires renderer, UI, and state together.
 *
 * Keeping this in one place prevents circular imports and makes dependencies explicit.
 */
import type { CliRenderer } from "@opentui/core"
import type { UI } from "./components/layout"

import { appState } from "./state/app-state"
import { config, pollingState } from "./state/polling"
import { snapshot } from "./state/snapshot"

export type RenderFn = () => void

export type AppContext = {
  renderer: CliRenderer
  ui: UI

  state: {
    snapshot: typeof snapshot
    appState: typeof appState
    pollingState: typeof pollingState
    config: typeof config
  }

  /** Triggers a full UI sync from state (implemented in ui/render.ts). */
  render: RenderFn
  /** Requests a render from the OpenTUI renderer. */
  requestRender: () => void
}

export function createAppContext(args: {
  renderer: CliRenderer
  ui: UI
  render: RenderFn
}): AppContext {
  const { renderer, ui, render } = args
  return {
    renderer,
    ui,
    state: {
      snapshot,
      appState,
      pollingState,
      config,
    },
    render,
    requestRender: () => renderer.requestRender(),
  }
}


