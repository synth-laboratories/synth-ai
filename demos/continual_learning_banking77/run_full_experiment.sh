#!/bin/bash
# =============================================================================
# Banking77 Continual Learning Full Experiment
# =============================================================================
# Runs all experiments (MIPRO Continual + Classic GEPA) and organizes results
#
# Configuration:
#   - Training: 100 rollouts per split
#   - Eval: 500 samples per split with high parallelization
#   - Model: gpt-4.1-nano
# =============================================================================

set -e  # Exit on error

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Configuration
TRAIN_ROLLOUTS=100
EVAL_SAMPLES=500
PARALLEL=50
MODEL="gpt-4.1-nano"
TRAIN_SIZE=30

# Create timestamped results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/experiment_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Log file for the full experiment
LOG_FILE="$RESULTS_DIR/experiment.log"

# Helper function to print and log
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Print header
log "============================================================================="
log "Banking77 Continual Learning Experiment"
log "============================================================================="
log "Started: $(date)"
log "Results directory: $RESULTS_DIR"
log ""
log "Configuration:"
log "  - Training rollouts per split: $TRAIN_ROLLOUTS"
log "  - Eval samples per split: $EVAL_SAMPLES"
log "  - Parallelization: $PARALLEL"
log "  - Model: $MODEL"
log "  - Train size: $TRAIN_SIZE"
log "============================================================================="
log ""

# Check for SYNTH_API_KEY
if [ -z "$SYNTH_API_KEY" ]; then
    log "ERROR: SYNTH_API_KEY environment variable not set"
    log "Please run: export SYNTH_API_KEY=sk_live_..."
    exit 1
fi

log "SYNTH_API_KEY is set (${SYNTH_API_KEY:0:12}...)"
log ""

# =============================================================================
# PHASE 1: MIPRO Continual Learning
# =============================================================================
log "============================================================================="
log "[PHASE 1/3] MIPRO Continual Learning (all 4 splits)"
log "============================================================================="
log ""

MIPRO_OUTPUT="$RESULTS_DIR/mipro_continual.json"
log "Running: uv run python run_mipro_continual.py"
log "  --rollouts-per-split $TRAIN_ROLLOUTS"
log "  --model $MODEL"
log "  --train-size $TRAIN_SIZE"
log "  --output $MIPRO_OUTPUT"
log ""

uv run python run_mipro_continual.py \
    --rollouts-per-split $TRAIN_ROLLOUTS \
    --model $MODEL \
    --train-size $TRAIN_SIZE \
    --output "$MIPRO_OUTPUT" \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "[PHASE 1/3] MIPRO Continual Learning COMPLETE"
log "  Output: $MIPRO_OUTPUT"
log ""

# =============================================================================
# PHASE 2: Classic GEPA (all splits)
# =============================================================================
log "============================================================================="
log "[PHASE 2/3] Classic GEPA (all 4 splits)"
log "============================================================================="
log ""

GEPA_OUTPUT="$RESULTS_DIR/classic_gepa.json"
log "Running: uv run python run_classic_gepa.py"
log "  --rollouts $TRAIN_ROLLOUTS"
log "  --model $MODEL"
log "  --train-size $TRAIN_SIZE"
log "  --output $GEPA_OUTPUT"
log ""

uv run python run_classic_gepa.py \
    --rollouts $TRAIN_ROLLOUTS \
    --model $MODEL \
    --train-size $TRAIN_SIZE \
    --output "$GEPA_OUTPUT" \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "[PHASE 2/3] Classic GEPA COMPLETE"
log "  Output: $GEPA_OUTPUT"
log ""

# =============================================================================
# PHASE 3: Held-Out Evaluation (500 samples, high parallelization)
# =============================================================================
log "============================================================================="
log "[PHASE 3/3] Held-Out Evaluation ($EVAL_SAMPLES samples per split)"
log "============================================================================="
log ""

EVAL_OUTPUT="$RESULTS_DIR/held_out_eval.json"
log "Running: uv run python run_held_out_eval.py"
log "  --eval-only"
log "  --mipro-results $MIPRO_OUTPUT"
log "  --gepa-results $GEPA_OUTPUT"
log "  --max-concurrent $PARALLEL"
log "  --output-dir $RESULTS_DIR"
log ""

uv run python run_held_out_eval.py \
    --eval-only \
    --mipro-results "$MIPRO_OUTPUT" \
    --gepa-results "$GEPA_OUTPUT" \
    --max-concurrent $PARALLEL \
    --output-dir "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "[PHASE 3/3] Held-Out Evaluation COMPLETE"
log ""

# =============================================================================
# Generate Summary
# =============================================================================
log "============================================================================="
log "EXPERIMENT COMPLETE"
log "============================================================================="
log ""
log "Results saved to: $RESULTS_DIR"
log ""
log "Files:"
ls -la "$RESULTS_DIR" | tee -a "$LOG_FILE"
log ""
log "Finished: $(date)"
log "============================================================================="

# Create a summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
{
    echo "Banking77 Continual Learning Experiment Summary"
    echo "================================================"
    echo ""
    echo "Configuration:"
    echo "  Training rollouts per split: $TRAIN_ROLLOUTS"
    echo "  Eval samples per split: $EVAL_SAMPLES"
    echo "  Model: $MODEL"
    echo "  Train size: $TRAIN_SIZE"
    echo "  Parallelization: $PARALLEL"
    echo ""
    echo "Timestamp: $TIMESTAMP"
    echo ""
    echo "Files:"
    echo "  - mipro_continual.json: MIPRO continual learning results"
    echo "  - classic_gepa.json: Classic GEPA results (cold/warm start)"
    echo "  - held_out_eval.json: Held-out test set evaluation"
    echo "  - experiment.log: Full experiment log"
    echo ""
    echo "To analyze results:"
    echo "  uv run python analyze_results.py --continual $MIPRO_OUTPUT --classic $GEPA_OUTPUT"
} > "$SUMMARY_FILE"

log ""
log "Summary saved to: $SUMMARY_FILE"
