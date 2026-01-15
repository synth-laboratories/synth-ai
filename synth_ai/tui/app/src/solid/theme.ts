/**
 * Shared visual styling constants for the SolidJS TUI.
 * Mirrored from the gold/reference theme.
 */

export const COLORS = {
  text: "#e2e8f0",
  textDim: "#94a3b8",
  textBright: "#f8fafc",
  textAccent: "#60a5fa",
  textSelected: "#ffffff",
  
  bg: "#0b1120",
  bgHeader: "#1e293b",
  bgTabs: "#111827",
  bgSelection: "#2563eb",
  bgSelectionFocused: "#1e293b",
  
  border: "#94a3b8",
  borderDim: "#1f2937",
  borderAccent: "#60a5fa",
  
  success: "#10b981",
  error: "#f87171",
  warning: "#fbbf24",
}

/** Default styling for boxes */
export const BOX = {
  borderStyle: "single" as const,
  borderColor: COLORS.border,
  borderColorFocused: COLORS.borderAccent,
  textColor: COLORS.text,
  backgroundColor: COLORS.bg,
}

/** Standard panel styling for detail panels */
export const PANEL = {
  border: true as const,
  borderStyle: "single" as const,
  borderColor: COLORS.border,
  borderColorFocused: COLORS.textAccent,
  paddingLeft: 1,
  titleAlignment: "left" as const,
}

/** Standard text styling */
export const TEXT = {
  /** Default text color for panel content */
  fg: COLORS.text,
  /** Dimmed text for secondary/hint content */
  fgDim: COLORS.textDim,
  /** Bright text for emphasis */
  fgBright: COLORS.textBright,
}

/** Get border color based on focus state */
export function getPanelBorderColor(focused: boolean): string {
  return focused ? PANEL.borderColorFocused : PANEL.borderColor
}


