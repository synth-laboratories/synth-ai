/**
 * Scanner for LocalAPI files in a directory.
 */
import * as fs from "fs"
import * as path from "path"

export type ScannedLocalAPI = {
	filename: string
	filepath: string
}

/**
 * Scan a directory for LocalAPI files.
 * Detection: file contains `from synth_ai.sdk.localapi import` or `create_local_api(`
 */
export function scanForLocalAPIs(directory: string): ScannedLocalAPI[] {
	const results: ScannedLocalAPI[] = []

	try {
		const entries = fs.readdirSync(directory, { withFileTypes: true })

		for (const entry of entries) {
			if (!entry.isFile() || !entry.name.endsWith(".py")) continue

			const filepath = path.join(directory, entry.name)
			try {
				const content = fs.readFileSync(filepath, "utf-8")
				if (isLocalAPIFile(content)) {
					results.push({
						filename: entry.name,
						filepath,
					})
				}
			} catch {
				// Skip files we can't read
			}
		}
	} catch {
		// Directory doesn't exist or can't be read
	}

	return results
}

/**
 * Check if file content indicates a LocalAPI file.
 */
function isLocalAPIFile(content: string): boolean {
	return (
		content.includes("from synth_ai.sdk.localapi import") ||
		content.includes("create_local_api(")
	)
}
