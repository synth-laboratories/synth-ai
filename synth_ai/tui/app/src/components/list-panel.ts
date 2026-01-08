/**
 * Generic list panel abstraction for the left sidebar.
 * Provides a reusable component for displaying selectable lists.
 */
import type { SelectRenderable, BoxRenderable, TextRenderable } from "@opentui/core"

// Re-export ListPanelId from types
export type { ListPanelId } from "../types"

/**
 * A formatted item ready for display in the list panel.
 */
export interface ListPanelItem {
  id: string
  name: string
  description: string
}

/**
 * Configuration for a list panel data source.
 */
export interface ListPanelConfig<T> {
  /** Unique identifier for this list type */
  id: string
  /** Title shown in the panel header */
  title: string
  /** Message shown when list is empty */
  emptyMessage: string
  /** Format a data item for display */
  formatItem: (item: T) => ListPanelItem
  /** Get the current list of items */
  getItems: () => T[]
  /** Called when an item is selected (can be async) */
  onSelect: (item: T, index: number) => void | Promise<void>
  /** Optional: Get title suffix (e.g., filter info) */
  getTitleSuffix?: () => string
}

/**
 * UI elements used by the list panel.
 */
export interface ListPanelUI {
  box: BoxRenderable
  select: SelectRenderable
  emptyText: TextRenderable
}

/**
 * Renders a list panel with the given configuration.
 */
export function renderListPanel<T>(
  ui: ListPanelUI,
  config: ListPanelConfig<T>
): void {
  const items = config.getItems()
  const suffix = config.getTitleSuffix?.() ?? ""
  ui.box.title = suffix ? `${config.title} (${suffix})` : config.title

  if (items.length) {
    ui.select.visible = true
    ui.emptyText.visible = false
    ui.select.options = items.map((item) => {
      const formatted = config.formatItem(item)
      return {
        name: formatted.name,
        description: formatted.description,
        value: formatted.id,
      }
    })
  } else {
    ui.select.visible = false
    ui.emptyText.visible = true
    ui.emptyText.content = config.emptyMessage
  }
}

