import { copyToClipboard } from "../utils/clipboard"
import { formatErrorMessage } from "../utils/truncate"
import { focusManager, type Focusable } from "../focus"

export type ErrorBoxOptions = {
	id: string
	defaultVisibleLines?: number
	maxWidth?: number
	onChange?: () => void
	onCopy?: (message: string) => void
}

type RenderOptions = {
	indent?: number
	maxWidth?: number
}

type HandleKeyOptions = {
	allowWhenNotFocused?: boolean
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(Math.max(value, min), max)
}

/**
 * Reusable error display box with scrolling + copy support.
 * Uses the global focus manager so it can temporarily steal focus
 * (Shift+Tab) without breaking existing key handling.
 */
export function createErrorBox(options: ErrorBoxOptions) {
	const state = {
		rawError: null as string | null,
		lines: [] as string[],
		offset: 0,
		visibleLines: options.defaultVisibleLines ?? 3,
		maxWidth: options.maxWidth ?? 64,
		focused: false,
		wrapWidth: Math.max(8, (options.maxWidth ?? 64) - 4),
	}

	const focusable: Focusable = {
		id: options.id,
		onFocus: () => {
			state.focused = true
			options.onChange?.()
		},
		onBlur: () => {
			state.focused = false
			options.onChange?.()
		},
		handleKey: (key: any) => handleKeyInternal(key, false),
	}

	function triggerUpdate(): void {
		options.onChange?.()
	}

	function rewrapIfNeeded(maxWidth?: number): void {
		if (!state.rawError) return
		const nextMaxWidth = Math.max(10, maxWidth ?? state.maxWidth)
		const wrapWidth = Math.max(4, nextMaxWidth - 4)
		if (wrapWidth !== state.wrapWidth) {
			state.wrapWidth = wrapWidth
			state.lines = formatErrorMessage(state.rawError, wrapWidth, Infinity)
			clampOffset()
		}
		state.maxWidth = nextMaxWidth
	}

	function clampOffset(): void {
		const maxOffset = Math.max(0, state.lines.length - state.visibleLines)
		state.offset = clamp(state.offset, 0, maxOffset)
	}

	function setError(error: string | null, override?: { visibleLines?: number; maxWidth?: number }): void {
		state.rawError = error
		state.visibleLines = override?.visibleLines ?? options.defaultVisibleLines ?? 3
		state.maxWidth = override?.maxWidth ?? options.maxWidth ?? state.maxWidth
		state.wrapWidth = Math.max(4, state.maxWidth - 4)
		state.offset = 0
		state.lines = error ? formatErrorMessage(error, state.wrapWidth, Infinity) : []
		if (!error && state.focused) {
			focusManager.pop(options.id)
		}
		triggerUpdate()
	}

	function clear(): void {
		setError(null)
	}

	function scroll(delta: number): void {
		if (!state.rawError) return
		const maxOffset = Math.max(0, state.lines.length - state.visibleLines)
		state.offset = clamp(state.offset + delta, 0, maxOffset)
		triggerUpdate()
	}

	async function copy(): Promise<void> {
		if (!state.rawError) return
		try {
			await copyToClipboard(state.rawError)
			options.onCopy?.(state.rawError)
		} catch {
			// Ignore copy errors
		}
	}

	function focus(): void {
		if (!state.rawError || state.focused) return
		focusManager.push(focusable)
	}

	function blur(): void {
		if (!state.focused) return
		focusManager.pop(options.id)
	}

	function handleKeyInternal(key: any, allowWhenNotFocused: boolean): boolean {
		if (!state.rawError) return false

		// Allow Shift-modified actions without stealing focus (for parents)
		if (!state.focused && allowWhenNotFocused) {
			if (key.shift && key.name === "tab") {
				focus()
				return true
			}
			if (key.shift && (key.name === "j" || key.name === "down")) {
				scroll(1)
				return true
			}
			if (key.shift && (key.name === "k" || key.name === "up")) {
				scroll(-1)
				return true
			}
			if (key.name === "y" || key.name === "c") {
				void copy()
				return true
			}
			return false
		}

		if (!state.focused) return false

		if (key.name === "tab") {
			blur()
			return true
		}
		if (key.name === "escape" || key.name === "q") {
			blur()
			return true
		}
		if (key.name === "up" || key.name === "k") {
			scroll(-1)
			return true
		}
		if (key.name === "down" || key.name === "j") {
			scroll(1)
			return true
		}
		if (key.name === "pageup") {
			scroll(-state.visibleLines)
			return true
		}
		if (key.name === "pagedown") {
			scroll(state.visibleLines)
			return true
		}
		if (key.name === "y" || key.name === "c") {
			void copy()
			return true
		}
		return false
	}

	function handleKey(key: any, opts?: HandleKeyOptions): boolean {
		return handleKeyInternal(key, opts?.allowWhenNotFocused ?? false)
	}

	function renderLines(opts?: RenderOptions): string[] {
		if (!state.rawError) return []

		const indent = " ".repeat(opts?.indent ?? 2)
		const maxWidth = Math.max(10, opts?.maxWidth ?? state.maxWidth)
		rewrapIfNeeded(maxWidth)

		const innerWidth = maxWidth - 2
		const contentWidth = Math.max(1, innerWidth - 2)
		const visible = state.lines.slice(state.offset, state.offset + state.visibleLines)
		const borderColor = state.focused ? "\x1b[91m" : "\x1b[90m"

		const lines: string[] = []
		lines.push(`${indent}${borderColor}+${"-".repeat(innerWidth)}+\x1b[0m\x1b[K`)
		for (let i = 0; i < state.visibleLines; i++) {
			const raw = visible[i] ?? ""
			// Slice to exact width and pad with spaces to ensure full overwrite
			const clipped = raw.slice(0, contentWidth)
			const padded = clipped.padEnd(contentWidth, " ")
			// Use \x1b[K (clear to end of line) to ensure no leftover characters from previous renders
			lines.push(`${indent}${borderColor}|\x1b[0m \x1b[31m${padded}\x1b[0m ${borderColor}|\x1b[0m\x1b[K`)
		}
		lines.push(`${indent}${borderColor}+${"-".repeat(innerWidth)}+\x1b[0m\x1b[K`)
		return lines
	}

	function getPositionLabel(): string | null {
		if (!state.rawError) return null
		const start = state.offset + 1
		const end = Math.min(state.offset + state.visibleLines, state.lines.length)
		return `[${start}-${end}/${state.lines.length}]`
	}

	function getHint(): string | null {
		if (!state.rawError) return null
		const position = getPositionLabel()
		const scrollHint = state.lines.length > state.visibleLines
			? state.focused ? "j/k scroll" : "shift+j/k scroll"
			: null
		const focusHint = state.focused ? "tab back" : "shift+tab focus"
		const copyHint = "y copy"
		return [position, scrollHint, copyHint, focusHint].filter(Boolean).join(" | ")
	}

	return {
		hasError: (): boolean => !!state.rawError,
		setError,
		clear,
		renderLines,
		handleKey,
		getHint,
		getPositionLabel,
		focus,
		blur,
		isFocused: (): boolean => state.focused,
	}
}
