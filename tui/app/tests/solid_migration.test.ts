import { describe, expect, test } from "bun:test"
import { resolveRendererMode, shouldUseSolidRenderer } from "../src/solid/mode"
import { computeLayoutMetrics, defaultLayoutSpec } from "../src/solid/layout"

describe("solid migration mode", () => {
  test("explicit renderer selection wins", () => {
    expect(resolveRendererMode({ SYNTH_TUI_RENDERER: "solid" })).toBe("solid")
    expect(resolveRendererMode({ SYNTH_TUI_RENDERER: "legacy" })).toBe("legacy")
  })

  test("solid flag enables solid renderer", () => {
    expect(shouldUseSolidRenderer({ SYNTH_TUI_SOLID: "true" })).toBe(true)
    expect(shouldUseSolidRenderer({ SYNTH_TUI_SOLID: "1" })).toBe(true)
  })

  test("legacy wins over solid flag", () => {
    expect(
      shouldUseSolidRenderer({ SYNTH_TUI_RENDERER: "legacy", SYNTH_TUI_SOLID: "1" }),
    ).toBe(false)
  })
})

describe("solid layout metrics", () => {
  test("computes baseline layout from terminal dimensions", () => {
    const metrics = computeLayoutMetrics(120, 40)
    expect(metrics.reservedHeight).toBe(
      defaultLayoutSpec.headerHeight
        + defaultLayoutSpec.tabsHeight
        + defaultLayoutSpec.statusHeight
        + defaultLayoutSpec.footerHeight,
    )
    expect(metrics.contentHeight).toBe(30)
    expect(metrics.jobsWidth).toBe(36)
    expect(metrics.detailWidth).toBe(84)
    expect(metrics.compact).toBe(false)
  })

  test("clamps when terminal is too small", () => {
    const metrics = computeLayoutMetrics(30, 8)
    expect(metrics.contentHeight).toBe(0)
    expect(metrics.jobsWidth).toBe(0)
    expect(metrics.detailWidth).toBe(30)
    expect(metrics.compact).toBe(true)
  })
})
