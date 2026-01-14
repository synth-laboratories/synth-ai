import { onCleanup } from "solid-js"
import type { SetStoreFunction } from "solid-js/store"

import type { AppData } from "../../types"
import { createAbortControllerRegistry, isAbortError } from "../../utils/abort"
import { registerCleanup, unregisterCleanup } from "../../lifecycle"
import { log, logError } from "../../utils/log"

type ActionTask = () => Promise<unknown> | void

export type ActionRunnerState = {
  actions: ReturnType<typeof createAbortControllerRegistry>
  runAction: (key: string, task: ActionTask) => void
}

type UseActionRunnerOptions = {
  setData: SetStoreFunction<AppData>
}

export function useActionRunner(options: UseActionRunnerOptions): ActionRunnerState {
  const actions = createAbortControllerRegistry()
  const cleanupName = "solid-actions-abort"
  registerCleanup(cleanupName, () => actions.abortAll())
  onCleanup(() => {
    actions.abortAll()
    unregisterCleanup(cleanupName)
  })

  const runAction = (key: string, task: ActionTask): void => {
    log("action", `runAction: starting ${key}`)
    void actions.run(key, () => task())
      .then(() => {
        log("action", `runAction: completed ${key}`)
      })
      .catch((err) => {
        if (isAbortError(err)) {
          log("action", `runAction: aborted ${key}`)
          return
        }
        logError(`runAction: ${key} failed`, err)
        options.setData("lastError", err?.message || "Action failed")
      })
  }

  return {
    actions,
    runAction,
  }
}
