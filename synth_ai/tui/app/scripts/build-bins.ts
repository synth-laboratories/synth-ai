import solidPlugin from "@opentui/solid/bun-plugin"
import { readdirSync, rmSync } from "node:fs"
import path from "node:path"

const external = [
  "@opentui/core",
  "@opentui/core-darwin-arm64",
  "@opentui/core-darwin-x64",
  "@opentui/core-linux-arm64",
  "@opentui/core-linux-x64",
  "@opentui/core-win32-arm64",
  "@opentui/core-win32-x64",
]

const targets = [
  "bun-darwin-arm64",
  "bun-darwin-x64",
  "bun-linux-arm64",
  "bun-linux-x64",
  "bun-windows-x64",
]

for (const target of targets) {
  const outfile = `synth-ai-tui-${target.replace(/^bun-/, "")}`
  console.log(`Building ${outfile}...`)
  const result = await Bun.build({
    entrypoints: ["./src/index.ts"],
    outdir: "./dist",
    target: "bun",
    format: "esm",
    plugins: [solidPlugin],
    minify: true,
    external,
    naming: {
      entry: "[dir]/[name].mjs",
      chunk: "[dir]/[name].mjs",
      asset: "[dir]/[name].[ext]",
    },
    compile: {
      target,
      outfile,
    },
  })

  if (!result.success) {
    for (const log of result.logs) {
      console.error(log.message)
    }
    process.exit(1)
  }
  console.log(`Built ${outfile}`)
}

for (const entry of readdirSync("./dist")) {
  if (entry.endsWith(".mjs")) {
    rmSync(path.join("./dist", entry))
  }
}
