# Judge Model Comparison Results

This directory contains performance comparison results for different judge models.

## Models Tested

1. **groq-qwen3-32b** - Groq's Qwen 3 32B model
2. **gpt-5-nano** - OpenAI's fastest model
3. **gpt-5-mini** - OpenAI's mid-tier model  
4. **gpt-5** - OpenAI's flagship model

## Metrics Captured

For each model, the following metrics are recorded:

### Performance
- **Total Wall Clock Time** - End-to-end test execution time
- **Evaluation Time** - Actual judge evaluation time
- **Traces Succeeded/Total** - Success rate

### Quality
- **Event Pearson Correlation** - How well event-level scores correlate with deterministic metrics
- **Outcome Pearson Correlation** - How well outcome-level scores correlate with deterministic metrics

### Efficiency
- **Avg API Call Time** - Average time per window evaluation
- **Avg Semaphore Wait** - Time spent waiting for concurrency slots
- **Semaphore Wait %** - Wait time as percentage of API time
- **Rate Limit Errors** - Number of 429 errors encountered

### Distribution
- **p0, p50, p90, p99, p100** - Completion time percentiles

## Files

- `00_comparison_summary.txt` - **START HERE** - Quick comparison table and analysis
- `{model_name}_results.txt` - Detailed results for each model including raw output

## Running the Test

```bash
# From synth-ai root
./scripts/run_judge_model_comparison.sh

# Or via pytest
uv run pytest tests/integration/test_judge_models_comparison.py -v -s
```

## Interpreting Results

### Speed
Lower total time = faster overall. Lower API call time = faster per-request.

### Quality
Higher Pearson correlation (closer to 1.0) = better alignment with deterministic metrics.
- `r > 0.7` = Strong correlation
- `0.3 < r < 0.7` = Moderate correlation
- `r < 0.3` = Weak correlation

### Efficiency
Lower semaphore wait % = less time spent queuing for API slots.
- `< 10%` = Very efficient, could increase concurrency
- `10-30%` = Well-balanced
- `> 50%` = High contention, may need more concurrency or is hitting rate limits

### Rate Limits
Zero 429 errors is ideal. If you see errors, reduce semaphore limit for that model in `backend/app/ai/judge/judge_semaphore.py`.


