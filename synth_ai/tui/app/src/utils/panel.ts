export function getPanelContentWidth(
  width: number,
  paddingLeft: number = 1,
  paddingRight: number = 0,
): number {
  return Math.max(1, Math.floor(width) - 2 - paddingLeft - paddingRight)
}

export function getPanelContentHeight(height: number): number {
  return Math.max(1, Math.floor(height) - 2)
}
