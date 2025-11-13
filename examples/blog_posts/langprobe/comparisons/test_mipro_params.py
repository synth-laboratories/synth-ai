#!/usr/bin/env python3
"""Example showing how to manually control MIPROv2 execution count."""

# Instead of using auto:
# optimizer = MIPROv2(metric=metric, auto="light")

# You can manually specify parameters:
optimizer = MIPROv2(
    metric=metric,
    num_candidates=10,           # Number of instruction candidates to generate
    init_temperature=1.0,        # Temperature for instruction generation
    # The main drivers of execution count:
    num_trials=20,               # Number of optimization trials (each trial evaluates on valset)
    # Each trial generates candidates and evaluates them
)

# Key parameters that affect execution count:
# - num_candidates: More candidates = more evaluations per trial
# - num_trials: More trials = more rounds of evaluation
# - valset size: Each candidate is evaluated on the entire valset
#
# Total evaluations ≈ num_candidates × num_trials × len(valset)
# Plus bootstrapping evaluations during demo generation
