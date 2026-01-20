export type LayoutSpec = {
  headerHeight: number
  tabsHeight: number
  statusHeight: number
  footerHeight: number
  jobsWidth: number
  minDetailWidth: number
  minContentHeight: number
}

export type LayoutMetrics = {
  totalWidth: number
  totalHeight: number
  reservedHeight: number
  contentHeight: number
  jobsWidth: number
  detailWidth: number
  compact: boolean
}

export const defaultLayoutSpec: LayoutSpec = {
  headerHeight: 3,
  tabsHeight: 2,
  statusHeight: 3,
  footerHeight: 2,
  jobsWidth: 36,
  minDetailWidth: 48,
  minContentHeight: 12,
}

export function computeLayoutMetrics(
  width: number,
  height: number,
  spec: LayoutSpec = defaultLayoutSpec,
): LayoutMetrics {
  const totalWidth = Math.max(0, Math.floor(width))
  const totalHeight = Math.max(0, Math.floor(height))
  const reservedHeight = spec.headerHeight + spec.tabsHeight + spec.statusHeight + spec.footerHeight
  const contentHeight = Math.max(0, totalHeight - reservedHeight)

  const minDetailWidth = Math.min(spec.minDetailWidth, totalWidth)
  const maxJobsWidth = Math.max(0, totalWidth - minDetailWidth)
  const jobsWidth = Math.min(spec.jobsWidth, maxJobsWidth)
  const detailWidth = Math.max(0, totalWidth - jobsWidth)

  return {
    totalWidth,
    totalHeight,
    reservedHeight,
    contentHeight,
    jobsWidth,
    detailWidth,
    compact: contentHeight < spec.minContentHeight || totalWidth < spec.minDetailWidth,
  }
}
