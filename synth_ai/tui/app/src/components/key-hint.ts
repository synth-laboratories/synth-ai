/**
 * Key hint primitive for consistent tab/shortcut styling.
 */
import { TextRenderable, type CliRenderer } from "@opentui/core"

type KeyHintOptions = {
	id: string
	description: string
	key: string
	active?: boolean // true = white (#f8fafc), false = gray (#94a3b8)
}

export function createKeyHint(
	renderer: CliRenderer,
	opts: KeyHintOptions,
): TextRenderable {
	return new TextRenderable(renderer, {
		id: opts.id,
		content: `${opts.description} (${opts.key})`,
		fg: opts.active ? "#f8fafc" : "#94a3b8",
	})
}
