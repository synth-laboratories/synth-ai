#!/usr/bin/env python3
"""
Focused test for the Synth backend that only covers:

1. Fine-tuning Qwen3-0.6B with a small JSONL dataset.
2. Running inference against the resulting fine-tuned model(s).

The test deliberately skips:
• DPO training
• Inference on the base (non-fine-tuned) model

It re-uses the SynthQwenTester implementation from `test_synth_qwen.py` to avoid
duplicating helper logic and keeps runtime to the minimum required to verify
fine-tune → inference functionality.
"""

import asyncio

# Re-use the full tester from the comprehensive test module.
from test_synth_qwen import SynthQwenTester


async def main() -> None:
    print("Synth Backend Fine-Tune-Only Test Suite")
    tester = SynthQwenTester()

    # 1️⃣ Fine-tune the base model (SFT)
    ft_ok = await tester.test_finetuning()
    if not ft_ok:
        tester.print_summary()
        return

    # 2️⃣ Inference using the resulting fine-tuned model(s)
    await tester.test_finetuned_inference()

    # Final summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
