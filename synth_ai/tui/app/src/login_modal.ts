/**
 * Login modal UI controller for the TUI.
 *
 * Handles the login modal state and interactions, delegating
 * the actual auth flow to auth.ts.
 */

import type { CliRenderer } from "@opentui/core"
import { runDeviceCodeAuth, type AuthStatus } from "./auth"
import { createModalUI, type ModalUI } from "./modals/base"
import { pollingState, clearJobsTimer, clearEventsTimer } from "./state/polling"
import { setLoggedOutMarker, clearLoggedOutMarker, saveApiKey, deleteSavedApiKey } from "./utils/logout-marker"
import { focusManager } from "./focus"
import { appState, frontendKeys, frontendKeySources, getFrontendUrlId } from "./state/app-state"

/**
 * Snapshot state for updating status messages.
 */
export type SnapshotState = {
  jobs: any[]
  selectedJob: any | null
  events: any[]
  metrics: Record<string, unknown>
  bestSnapshotId: string | null
  bestSnapshot: Record<string, any> | null
  evalSummary: Record<string, any> | null
  evalResultRows: Array<Record<string, any>>
  artifacts: Array<Record<string, unknown>>
  orgId: string | null
  userId: string | null
  balanceDollars: number | null
  lastError: string | null
  status: string
  lastRefresh: number | null
  allCandidates: any[]
  apiCandidates: any[]
  apiCandidatesLoaded: boolean
}

/**
 * Dependencies required by the login modal controller.
 */
export type LoginModalDeps = {
  renderer: CliRenderer
  bootstrap: () => Promise<void>
  getSnapshot: () => SnapshotState
  renderSnapshot: () => void
}

/**
 * Login modal controller interface.
 */
export type LoginModalController = {
  /** Whether the login modal is currently visible */
  readonly isVisible: boolean
  /** Whether an auth flow is in progress */
  readonly isInProgress: boolean
  /** Current auth status */
  readonly status: AuthStatus
  /** Toggle the login modal visibility */
  toggle: (visible: boolean) => void
  /** Handle key input when modal is visible */
  handleKey: (key: any) => boolean
  /** Start the device code auth flow */
  startAuth: () => Promise<void>
  /** Log out */
  logout: () => Promise<void>
}

/**
 * Create a login modal controller with the given dependencies.
 */
export function createLoginModal(deps: LoginModalDeps): LoginModalController {
  let loginAuthStatus: AuthStatus = { state: "idle" }
  let loginAuthInProgress = false

  const { renderer, bootstrap, getSnapshot, renderSnapshot } = deps

  // Create modal UI using the primitive
  const modal: ModalUI = createModalUI(renderer, {
    id: "login-modal",
    width: 60,
    height: 10,
    borderColor: "#22c55e",
    titleColor: "#22c55e",
    zIndex: 15,
  })

  function updateLoginModalStatus(status: AuthStatus): void {
    loginAuthStatus = status
    switch (status.state) {
      case "idle":
        modal.setContent("Press Enter to open browser and sign in...")
        modal.setHint("Enter start | q cancel")
        break
      case "initializing":
        modal.setContent("Initializing...")
        modal.setHint("Please wait...")
        break
      case "waiting":
        modal.setContent([
          "Browser opened. Complete sign-in there.",
          "",
          `URL: ${status.verificationUri}`,
        ].join("\n"))
        modal.setHint("Waiting for browser auth... | q cancel")
        break
      case "polling":
        modal.setContent([
          "Browser opened. Complete sign-in there.",
          "",
          "Checking for completion...",
        ].join("\n"))
        modal.setHint("Waiting for browser auth... | q cancel")
        break
      case "success":
        modal.setContent("Authentication successful!")
        modal.setHint("Loading...")
        break
      case "error":
        modal.setContent(`Error: ${status.message}`)
        modal.setHint("Enter retry | q close")
        break
    }
    renderer.requestRender()
  }

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "login-modal",
        handleKey,
      })
      modal.center()
      loginAuthStatus = { state: "idle" }
      loginAuthInProgress = false
      modal.setTitle("Sign In / Sign Up")
      modal.setContent("Press Enter to open browser")
      modal.setHint("Enter start | q cancel")
    } else {
      focusManager.pop("login-modal")
    }
    modal.setVisible(visible)
  }

  async function startAuth(): Promise<void> {
    if (loginAuthInProgress) return
    loginAuthInProgress = true

    const result = await runDeviceCodeAuth(updateLoginModalStatus)

    loginAuthInProgress = false

    if (result.success && result.apiKey) {
      // Store the key by frontend URL (dev and local share localhost:3000)
      const frontendUrlId = getFrontendUrlId(appState.currentBackend)
      frontendKeys[frontendUrlId] = result.apiKey
      frontendKeySources[frontendUrlId] = { sourcePath: null, varName: "device_code_auth" }

      // Store the key in memory and persist to file
      process.env.SYNTH_API_KEY = result.apiKey
      await saveApiKey(result.apiKey)

      // Persist the key to settings file
      const { persistSettings } = await import("./persistence/settings")
      const { config } = await import("./state/polling")
      await persistSettings({
        settingsFilePath: config.settingsFilePath,
        getCurrentBackend: () => appState.currentBackend,
        getFrontendKey: (id) => frontendKeys[id],
        getFrontendKeySource: (id) => frontendKeySources[id],
      })

      // Clear logout marker so auto-login works next time
      await clearLoggedOutMarker()

      // Close modal and refresh
      toggle(false)
      const snapshot = getSnapshot()
      snapshot.lastError = null
      snapshot.status = "Authenticated! Loading..."
      renderSnapshot()

      // Bootstrap the app (loads jobs, identity, starts polling)
      await bootstrap()
    }
  }

  async function logout(): Promise<void> {
    // Mark as logged out and delete saved key
    await setLoggedOutMarker()
    await deleteSavedApiKey()

    process.env.SYNTH_API_KEY = ""

    // Disconnect SSE
    if (pollingState.sseDisconnect) {
      pollingState.sseDisconnect()
      pollingState.sseDisconnect = null
    }
    pollingState.sseConnected = false

    // Clear polling timers
    clearJobsTimer()
    clearEventsTimer()

    // Clear ALL auth-related state immediately
    const snapshot = getSnapshot()
    snapshot.jobs = []
    snapshot.selectedJob = null
    snapshot.events = []
    snapshot.metrics = {}
    snapshot.bestSnapshotId = null
    snapshot.bestSnapshot = null
    snapshot.evalSummary = null
    snapshot.evalResultRows = []
    snapshot.artifacts = []
    snapshot.orgId = null
    snapshot.userId = null
    snapshot.balanceDollars = null
    snapshot.lastRefresh = null
    snapshot.allCandidates = []
    snapshot.apiCandidates = []
    snapshot.apiCandidatesLoaded = false
    snapshot.lastError = "Logged out"
    snapshot.status = "Sign in required"
    renderSnapshot()

    // Show login modal
    toggle(true)
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false

    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    if (key.name === "return" || key.name === "enter") {
      void startAuth()
      return true
    }
    return true // consume all keys when modal is open
  }

  const controller = {
    get isVisible() {
      return modal.visible
    },
    get isInProgress() {
      return loginAuthInProgress
    },
    get status() {
      return loginAuthStatus
    },
    toggle,
    handleKey,
    startAuth,
    logout,
  }

  return controller
}
