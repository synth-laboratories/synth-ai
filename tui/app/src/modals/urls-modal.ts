/**
 * URLs modal controller - hidden dev feature showing backend/frontend URLs.
 */
import type { CliRenderer } from "@opentui/core"
import { createModalUI, type ModalController, type ModalUI } from "./base"
import { focusManager } from "../focus"

export function createUrlsModal(renderer: CliRenderer): ModalController & {
	open: () => void
} {
	const modal: ModalUI = createModalUI(renderer, {
		id: "urls-modal",
		width: 60,
		height: 10,
		borderColor: "#f59e0b",
		titleColor: "#f59e0b",
		zIndex: 10,
	})

	modal.setTitle("URLs")
	modal.setHint("q close")

	function updateContent(): void {
		const backend = process.env.SYNTH_BACKEND_URL || "-"
		const frontend = process.env.SYNTH_FRONTEND_URL || "-"
		modal.setContent(`Backend:\n${backend}\n\nFrontend:\n${frontend}`)
	}

	function toggle(visible: boolean): void {
		if (visible) {
			focusManager.push({
				id: "urls-modal",
				handleKey,
			})
			modal.center()
			updateContent()
		} else {
			focusManager.pop("urls-modal")
		}
		modal.setVisible(visible)
	}

	function open(): void {
		toggle(true)
	}

	function handleKey(key: any): boolean {
		if (!modal.visible) return false

		if (key.name === "return" || key.name === "enter" || key.name === "q" || key.name === "escape") {
			toggle(false)
			return true
		}
		return true
	}

	const controller = {
		get isVisible() {
			return modal.visible
		},
		toggle,
		open,
		handleKey,
	}

	return controller
}
