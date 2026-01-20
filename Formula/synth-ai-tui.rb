class SynthAiTui < Formula
  desc "Synth AI TUI - Terminal interface for Synth AI"
  homepage "https://github.com/synth-laboratories/synth-ai"
  version "0.1.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/synth-laboratories/synth-ai/releases/download/tui-v0.1.0/synth-ai-tui-darwin-arm64.tar.gz"
      sha256 "277f25680e468af9c18f9a27ba6a3f603ef3a15e601ae10cb4c29471d8032eca"
    else
      url "https://github.com/synth-laboratories/synth-ai/releases/download/tui-v0.1.0/synth-ai-tui-darwin-x64.tar.gz"
      sha256 "4769fcf94147cf242bab08ae3ac2da62ddc148a71dba4ad76f9ab056a01d74d0"
    end
  end

  def install
    bin.install "synth-ai-tui"
  end

  test do
    assert_match "synth", shell_output("#{bin}/synth-ai-tui --help 2>&1", 1)
  end
end
