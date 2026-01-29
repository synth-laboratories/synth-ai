/**
 * Auth state persistence utilities.
 * Handles logout marker and API key storage across TUI restarts.
 */
import fs from "node:fs"
import path from "node:path"
import os from "node:os"

const STATE_DIR = path.join(os.homedir(), ".synth-ai", ".tui")
const MARKER_PATH = path.join(STATE_DIR, "logged-out")
const API_KEY_PATH = path.join(STATE_DIR, "api-key")

/**
 * Check if the logout marker file exists (sync for startup).
 */
export function isLoggedOutMarkerSet(): boolean {
	try {
		fs.accessSync(MARKER_PATH, fs.constants.F_OK)
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
		await fs.promises.mkdir(STATE_DIR, { recursive: true })
		await fs.promises.writeFile(MARKER_PATH, "", "utf8")
	} catch {
		// Silent fail - not critical
	}
}

/**
 * Remove the logout marker file.
 */
export async function clearLoggedOutMarker(): Promise<void> {
	try {
		await fs.promises.unlink(MARKER_PATH)
	} catch {
		// Silent fail - file may not exist
	}
}

/**
 * Load saved API key from file (sync for startup).
 */
export function loadSavedApiKey(): string | null {
	try {
		const key = fs.readFileSync(API_KEY_PATH, "utf8").trim()
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
		await fs.promises.mkdir(STATE_DIR, { recursive: true })
		await fs.promises.writeFile(API_KEY_PATH, key, { encoding: "utf8", mode: 0o600 })
	} catch {
		// Silent fail - not critical
	}
}

/**
 * Delete saved API key file.
 */
export async function deleteSavedApiKey(): Promise<void> {
	try {
		await fs.promises.unlink(API_KEY_PATH)
	} catch {
		// Silent fail - file may not exist
	}
}
