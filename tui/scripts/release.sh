#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_dir="$root_dir/app"
dist_dir="$app_dir/dist"
release_dir="$root_dir/release"

mkdir -p "$release_dir"
rm -f "$release_dir"/synth-ai-tui-*.tar.gz "$release_dir/SHA256SUMS"

pushd "$app_dir" >/dev/null
bun install
bun run scripts/build-bins.ts
popd >/dev/null

for arch in darwin-arm64 darwin-x64; do
  binary="$dist_dir/synth-ai-tui-${arch}"
  if [[ ! -f "$binary" ]]; then
    echo "Missing build artifact: $binary" >&2
    exit 1
  fi

  tmp_dir="$(mktemp -d)"
  cp "$binary" "$tmp_dir/synth-ai-tui"
  chmod +x "$tmp_dir/synth-ai-tui"
  tar -C "$tmp_dir" -czf "$release_dir/synth-ai-tui-${arch}.tar.gz" synth-ai-tui
  rm -rf "$tmp_dir"
done

shasum -a 256 "$release_dir"/synth-ai-tui-*.tar.gz > "$release_dir/SHA256SUMS"

echo "Release artifacts ready in $release_dir"
