/**
 * Login modal UI controller for the TUI.
 *
 * Handles the login modal state and interactions, delegating
 * the actual auth flow to auth.ts.
 */

import {
  runDeviceCodeAuth,
  type AuthStatus,
  type BackendId,
} from "./auth"

/**
 * UI elements required by the login modal.
 * Using `any` for content since it can be StyledText or string depending on OpenTUI version.
 */
export type LoginModalUI = {
  loginModalVisible: boolean
  loginModalBox: { visible: boolean; left: number; top: number; width: number; height: number }
  loginModalTitle: { visible: boolean; content: any; left: number; top: number }
  loginModalText: { visible: boolean; content: any; left: number; top: number }
  loginModalHelp: { visible: boolean; content: any; left: number; top: number }
  jobsSelect: { blur: () => void; focus: () => void }
}

/**
 * Backend key management functions.
 */
export type BackendKeySource = {
  sourcePath: string | null
  varName: string | null
}

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
}

/**
 * Dependencies required by the login modal controller.
 */
export type LoginModalDeps = {
  ui: LoginModalUI
  renderer: { requestRender: () => void }
  getCurrentBackend: () => BackendId
  getBackendConfig: () => { label: string }
  getBackendKeys: () => Record<BackendId, string>
  setBackendKey: (backend: BackendId, key: string, source: BackendKeySource) => void
  persistSettings: () => Promise<void>
  bootstrap: () => Promise<void>
  getSnapshot: () => SnapshotState
  renderSnapshot: () => void
  getActivePane: () => "jobs" | "events"
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
  /** Start the device code auth flow */
  startAuth: () => Promise<void>
  /** Log out from the current backend */
  logout: () => Promise<void>
}

/**
 * Create a login modal controller with the given dependencies.
 */
export function createLoginModal(deps: LoginModalDeps): LoginModalController {
  let loginModalVisible = false
  let loginAuthStatus: AuthStatus = { state: "idle" }
  let loginAuthInProgress = false

  const {
    ui,
    renderer,
    getCurrentBackend,
    getBackendConfig,
    setBackendKey,
    persistSettings,
    bootstrap,
    getSnapshot,
    renderSnapshot,
    getActivePane,
  } = deps

  function updateUIVisibility(visible: boolean): void {
    loginModalVisible = visible
    ui.loginModalVisible = visible
    ui.loginModalBox.visible = visible
    ui.loginModalTitle.visible = visible
    ui.loginModalText.visible = visible
    ui.loginModalHelp.visible = visible
  }

  function updateLoginModalStatus(status: AuthStatus): void {
    loginAuthStatus = status
    switch (status.state) {
      case "idle":
        ui.loginModalText.content = "Press Enter to open browser and sign in..."
        ui.loginModalHelp.content = "Enter start | q cancel"
        break
      case "initializing":
        ui.loginModalText.content = "Initializing..."
        ui.loginModalHelp.content = "Please wait..."
        break
      case "waiting":
        ui.loginModalText.content = [
          "Browser opened. Complete sign-in there.",
          "",
          `URL: ${status.verificationUri}`,
        ].join("\n")
        ui.loginModalHelp.content = "Waiting for browser auth... | q cancel"
        break
      case "polling":
        ui.loginModalText.content = [
          "Browser opened. Complete sign-in there.",
          "",
          "Checking for completion...",
        ].join("\n")
        ui.loginModalHelp.content = "Waiting for browser auth... | q cancel"
        break
      case "success":
        ui.loginModalText.content = "Authentication successful!"
        ui.loginModalHelp.content = "Loading..."
        break
      case "error":
        ui.loginModalText.content = `Error: ${status.message}`
        ui.loginModalHelp.content = "Enter retry | q close"
        break
    }
    renderer.requestRender()
  }

  function toggle(visible: boolean): void {
    updateUIVisibility(visible)
    if (visible) {
      // Center the modal on screen
      const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
      const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
      const width = 60
      const height = 10
      const left = Math.max(0, Math.floor((cols - width) / 2))
      const top = Math.max(1, Math.floor((rows - height) / 2))

      ui.loginModalBox.left = left
      ui.loginModalBox.top = top
      ui.loginModalBox.width = width
      ui.loginModalBox.height = height
      ui.loginModalTitle.left = left + 2
      ui.loginModalTitle.top = top + 1
      ui.loginModalText.left = left + 2
      ui.loginModalText.top = top + 3
      ui.loginModalHelp.left = left + 2
      ui.loginModalHelp.top = top + height - 2

      loginAuthStatus = { state: "idle" }
      loginAuthInProgress = false
      ui.loginModalTitle.content = `Sign In / Sign Up`
      ui.loginModalText.content = "Press Enter to open browser"
      ui.loginModalHelp.content = "Enter start | q cancel"
      ui.jobsSelect.blur()
    } else {
      if (getActivePane() === "jobs") {
        ui.jobsSelect.focus()
      }
    }
    renderer.requestRender()
  }

  async function startAuth(): Promise<void> {
    if (loginAuthInProgress) return
    loginAuthInProgress = true

    const currentBackend = getCurrentBackend()
    const result = await runDeviceCodeAuth(currentBackend, updateLoginModalStatus)

    loginAuthInProgress = false

    if (result.success && result.apiKey) {
      // Store the key
      setBackendKey(currentBackend, result.apiKey, {
        sourcePath: "browser-auth",
        varName: null,
      })

      // Persist settings
      await persistSettings()

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
    const currentBackend = getCurrentBackend()
    setBackendKey(currentBackend, "", { sourcePath: null, varName: null })
    await persistSettings()

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
    snapshot.lastError = `Logged out from ${getBackendConfig().label}`
    snapshot.status = "Sign in required"
    renderSnapshot()

    // Show login modal
    toggle(true)
  }

  return {
    get isVisible() {
      return loginModalVisible
    },
    get isInProgress() {
      return loginAuthInProgress
    },
    get status() {
      return loginAuthStatus
    },
    toggle,
    startAuth,
    logout,
  }
}

