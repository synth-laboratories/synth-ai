import { createSignal, onMount, type Accessor } from "solid-js"
import type { SetStoreFunction } from "solid-js/store"

import type { AppData } from "../../types"
import type { AppState } from "../../state/app-state"
import type { ActiveModal } from "../modals/types"
import type { AuthStatus } from "../../auth"
import { runDeviceCodeAuth } from "../../auth"
import { persistModeKey, persistModeSelection } from "../../persistence/settings"
import { pollingState, clearEventsTimer, clearJobsTimer, resetSseConnections } from "../../state/polling"
import { createInitialData } from "../store"
type AbortRegistry = { abortAll: () => void }

type UseAuthFlowOptions = {
  ui: AppState
  data: AppData
  setData: SetStoreFunction<AppData>
  actions: AbortRegistry
  openOverlayModal: (kind: ActiveModal) => void
  closeActiveModal: () => void
  refreshData: () => Promise<boolean>
}

export type AuthFlowState = {
  loginStatus: Accessor<AuthStatus>
  startLoginAuth: () => Promise<void>
  logout: () => Promise<void>
  promptLogin: () => void
}

export function useAuthFlow(options: UseAuthFlowOptions): AuthFlowState {
  const [loginStatus, setLoginStatus] = createSignal<AuthStatus>({ state: "idle" })
  const [loginInProgress, setLoginInProgress] = createSignal(false)

  const promptLogin = (): void => {
    setLoginStatus({ state: "idle" })
    setLoginInProgress(false)
    options.openOverlayModal("login")
  }

  onMount(() => {
    if (!process.env.SYNTH_API_KEY && options.data.status === "Sign in required") {
      promptLogin()
    }
  })

  const startLoginAuth = async (): Promise<void> => {
    if (loginInProgress()) return
    setLoginInProgress(true)
    const result = await runDeviceCodeAuth((status) => {
      setLoginStatus(status)
    })
    setLoginInProgress(false)

    if (result.success && result.apiKey) {
      process.env.SYNTH_API_KEY = result.apiKey
      await persistModeKey(options.ui.currentMode, result.apiKey)
      if (options.ui.settingsMode) {
        await persistModeSelection(options.ui.settingsMode)
      }
      options.closeActiveModal()
      options.setData("lastError", null)
      options.setData("status", "Authenticated! Loading...")
      await options.refreshData()
    }
  }

  const logout = async (): Promise<void> => {
    options.actions.abortAll()
    process.env.SYNTH_API_KEY = ""
    await persistModeKey(options.ui.currentMode, "")

    if (pollingState.sseDisconnect) {
      pollingState.sseDisconnect()
      pollingState.sseDisconnect = null
    }
    resetSseConnections()
    clearJobsTimer()
    clearEventsTimer()

    const reset = createInitialData()
    reset.lastError = "Logged out"
    reset.status = "Sign in required"
    options.setData(reset)

    setLoginStatus({ state: "idle" })
    setLoginInProgress(false)
    options.openOverlayModal("login")
  }

  return {
    loginStatus,
    startLoginAuth,
    logout,
    promptLogin,
  }
}
