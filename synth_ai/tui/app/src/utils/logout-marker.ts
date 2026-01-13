/**
 * Auth state persistence utilities.
 * Handles logout marker and API key storage across TUI restarts.
 */
import fs from "node:fs"
import { tuiApiKeyPath, tuiConfigDir, tuiLoggedOutMarkerPath } from "../paths"

/**
 * Check if the logout marker file exists (sync for startup).
 */
export function isLoggedOutMarkerSet(): boolean {
	try {
    fs.accessSync(tuiLoggedOutMarkerPath, fs.constants.F_OK)
		return true
	} catch {
		return false
	}
}

/**
 * Create the logout marker file.
 */
export async function setLoggedOutMarker(): Promise<void> {
	try {
    await fs.promises.mkdir(tuiConfigDir, { recursive: true })
    await fs.promises.writeFile(tuiLoggedOutMarkerPath, "", "utf8")
	} catch {
		// Silent fail - not critical
	}
}

/**
 * Remove the logout marker file.
 */
export async function clearLoggedOutMarker(): Promise<void> {
	try {
    await fs.promises.unlink(tuiLoggedOutMarkerPath)
	} catch {
		// Silent fail - file may not exist
	}
}

/**
 * Load saved API key from file (sync for startup).
 */
export function loadSavedApiKey(): string | null {
	try {
    const key = fs.readFileSync(tuiApiKeyPath, "utf8").trim()
		return key || null
	} catch {
		return null
	}
}

/**
 * Save API key to file.
 */
export async function saveApiKey(key: string): Promise<void> {
	try {
    await fs.promises.mkdir(tuiConfigDir, { recursive: true })
    await fs.promises.writeFile(tuiApiKeyPath, key, { encoding: "utf8", mode: 0o600 })
	} catch {
		// Silent fail - not critical
	}
}

/**
 * Delete saved API key file.
 */
export async function deleteSavedApiKey(): Promise<void> {
	try {
    await fs.promises.unlink(tuiApiKeyPath)
	} catch {
		// Silent fail - file may not exist
	}
}
