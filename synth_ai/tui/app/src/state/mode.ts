/**
 * Mode configuration and switching logic.
 *
 * Handles prod/dev/local mode URLs and API keys.
 * Process env overrides are honored at startup.
 */

import { parseMode } from "../types"
import type { Mode, ModeUrls } from "../types"

/** Hardcoded default URLs for each mode */
const DEFAULT_URLS: Record<Mode, ModeUrls> = {
  prod: {
    backendUrl: "https://api.usesynth.ai/api",
    frontendUrl: "https://usesynth.ai",
  },
  dev: {
    backendUrl: "https://synth-backend-dev-docker.onrender.com/api",
    frontendUrl: "http://localhost:3000",
  },
  local: {
    backendUrl: "http://localhost:8000/api",
    frontendUrl: "http://localhost:3000",
  },
}

/** URLs for each mode */
export const modeUrls: Record<Mode, ModeUrls> = { ...DEFAULT_URLS }

/** API keys per mode */
export const modeKeys: Record<Mode, string> = {
  prod: "",
  dev: "",
  local: "",
}

/** Whether URLs were set via process.env at startup */
let envUrlsAtStartup = false

/** Current mode */
let currentMode: Mode = "prod"

/**
 * Initialize mode state from environment.
 * Call this once at startup.
 */
export function initModeState(): void {
  // Check if URLs were provided via process.env
  const envBackend = process.env.SYNTH_BACKEND_URL
  const envFrontend = process.env.SYNTH_FRONTEND_URL

  if (envBackend || envFrontend) {
    envUrlsAtStartup = true
  }

  // Determine initial mode
  const envMode = process.env.SYNTH_TUI_MODE
  if (envMode) {
    currentMode = parseMode(envMode)
  }

  // Initialize API key for current mode from env
  if (process.env.SYNTH_API_KEY) {
    modeKeys[currentMode] = process.env.SYNTH_API_KEY
  }

  // If no env URLs were set, apply the default URLs for the current mode
  if (!envUrlsAtStartup) {
    const urls = modeUrls[currentMode]
    process.env.SYNTH_BACKEND_URL = urls.backendUrl
    process.env.SYNTH_FRONTEND_URL = urls.frontendUrl
  }
}

/**
 * Get the current mode.
 */
export function getCurrentMode(): Mode {
  return currentMode
}

/**
 * Check if env URLs were set at startup.
 */
export function hasEnvUrlsAtStartup(): boolean {
  return envUrlsAtStartup
}

/**
 * Switch to a different mode.
 * Updates process.env URLs and API key.
 */
export function switchMode(mode: Mode): void {
  currentMode = mode
  process.env.SYNTH_TUI_MODE = mode

  // Always update URLs on explicit mode switch
  const urls = modeUrls[mode]
  process.env.SYNTH_BACKEND_URL = urls.backendUrl
  process.env.SYNTH_FRONTEND_URL = urls.frontendUrl
  process.env.SYNTH_API_KEY = modeKeys[mode] || ""
}

/**
 * Set the API key for a specific mode.
 */
export function setModeKey(mode: Mode, key: string): void {
  modeKeys[mode] = key
}

/**
 * Set the current mode without switching URLs.
 * Used when loading persisted settings.
 */
export function setCurrentMode(mode: Mode): void {
  currentMode = mode
}
