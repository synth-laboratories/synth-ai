/**
 * Unified Focus Manager
 *
 * Single source of truth for focus state across the entire TUI.
 * Uses a stack-based approach: when a modal opens, it pushes onto the stack
 * and auto-blurs the previous focused item. When it closes, it pops and
 * auto-restores the previous focus.
 */

/** Any focusable component (modal, pane, widget) */
export type Focusable = {
	id: string
	/** Called when this item receives focus */
	onFocus?: () => void
	/** Called when this item loses focus */
	onBlur?: () => void
	/** Handle keypress. Return true if consumed. */
	handleKey?: (key: any) => boolean
}

class FocusManager {
	private stack: Focusable[] = []
	private defaultFocusable: Focusable | null = null

	/** Set the default focusable (e.g., jobs pane). Called on startup. */
	setDefault(focusable: Focusable): void {
		this.defaultFocusable = focusable
		if (this.stack.length === 0) {
			focusable.onFocus?.()
		}
	}

	/** Push a new focusable onto the stack (auto-blurs current) */
	push(focusable: Focusable): void {
		// Blur current top of stack or default
		const current = this.current() ?? this.defaultFocusable
		current?.onBlur?.()

		this.stack.push(focusable)
		focusable.onFocus?.()
	}

	/** Pop focusable from stack (auto-restores previous or default) */
	pop(id?: string): void {
		if (id) {
			// Remove specific item by id
			const idx = this.stack.findIndex((f) => f.id === id)
			if (idx >= 0) {
				const removed = this.stack.splice(idx, 1)[0]
				removed?.onBlur?.()
			}
		} else {
			// Pop top of stack
			const removed = this.stack.pop()
			removed?.onBlur?.()
		}

		// Restore focus to new top or default
		const next = this.current() ?? this.defaultFocusable
		next?.onFocus?.()
	}

	/** Get current focused item (top of stack, or null if empty) */
	current(): Focusable | null {
		return this.stack[this.stack.length - 1] ?? null
	}

	/** Route keypress to current focused item. Returns true if consumed. */
	handleKey(key: any): boolean {
		const active = this.current()
		if (active?.handleKey) {
			return active.handleKey(key)
		}
		return false
	}

	/** Check if any overlay (modal) is on the stack */
	hasOverlay(): boolean {
		return this.stack.length > 0
	}

	/** Clear all items from the stack (useful for reset) */
	clear(): void {
		while (this.stack.length > 0) {
			const removed = this.stack.pop()
			removed?.onBlur?.()
		}
		this.defaultFocusable?.onFocus?.()
	}
}

export const focusManager = new FocusManager()
