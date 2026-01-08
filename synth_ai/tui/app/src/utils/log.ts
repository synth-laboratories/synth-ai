/**
 * Debug logging utility - writes to /tmp/tui.log
 */
import { appendFileSync } from "fs"

const LOG_FILE = "/tmp/tui.log"

export function log(...args: any[]): void {
	const ts = new Date().toISOString()
	const msg = args
		.map((a) => (typeof a === "object" ? JSON.stringify(a, null, 2) : String(a)))
		.join(" ")
	appendFileSync(LOG_FILE, `[${ts}] ${msg}\n`)
}
