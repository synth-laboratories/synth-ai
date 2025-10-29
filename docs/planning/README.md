# Planning & Design Documents

This directory contains planning documents, architecture designs, and analysis for major features and improvements to the Synth AI SDK.

## Documents

### Job Polling & Streaming

1. **`job_polling_analysis.md`** - Comprehensive analysis of the current job status polling logic for SFT and RL training
   - Current polling implementation in SDK and backend
   - Backend status generation and event emission
   - Data flow diagrams
   - Identified gaps and improvement opportunities
   - Live capture of actual SFT polling output

2. **`streaming_abstractions_design.md`** - Design for flexible, configurable job streaming
   - Core abstractions (`StreamType`, `StreamMessage`, `StreamConfig`, `StreamHandler`)
   - Built-in handlers (CLI, Rich, JSON, Callback)
   - JobStreamer multiplexer architecture
   - CLI and SDK integration examples
   - Implementation plan with phased rollout
   - Backward compatibility strategy

3. **`sft_polling_analysis.txt`** - Focused analysis of SFT polling output
   - What's currently shown vs. what's available
   - Comparison of current vs. ideal output
   - Specific recommendations for CLI improvements

## Status

These documents are **approved designs** ready for implementation. They represent the planned approach for:
- Upgrading the CLI train command to show rich training progress
- Providing flexible streaming abstractions for programmatic SDK usage
- Maintaining backward compatibility while adding powerful new features

## Related Code

- Current implementation: `synth_ai/api/train/pollers.py`, `synth_ai/learning/jobs.py`
- Backend emission: `monorepo/backend/app/orchestration/jobs/postgrest_emitter.py`
- CLI integration: `synth_ai/api/train/cli.py`

## Next Steps

See the implementation plan in `streaming_abstractions_design.md` for phased rollout:
1. Phase 1: Core abstractions (Week 1)
2. Phase 2: JobStreamer (Week 2)
3. Phase 3: Rich handlers (Week 3)
4. Phase 4: CLI integration (Week 4)
5. Phase 5: Testing & polish (Week 5)

