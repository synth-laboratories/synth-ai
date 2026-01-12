import solidPlugin from "@opentui/solid/bun-plugin"

const result = await Bun.build({
  entrypoints: ["./src/index.ts"],
  outdir: "./dist",
  target: "bun",
  format: "esm",
  plugins: [solidPlugin],
  minify: true,
  naming: {
    entry: "[dir]/[name].mjs",
    chunk: "[dir]/[name].mjs",
    asset: "[dir]/[name].[ext]",
  },
})

if (!result.success) {
  for (const log of result.logs) {
    console.error(log.message)
  }
  process.exit(1)
}
