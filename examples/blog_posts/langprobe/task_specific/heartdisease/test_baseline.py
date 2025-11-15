#!/usr/bin/env python3
"""Test baseline evaluation to debug 0.0 scores."""

import os
from dotenv import load_dotenv
load_dotenv()

from dspy_heartdisease_adapter import (
    load_heartdisease_dataset,
    create_dspy_examples,
    heartdisease_metric,
    HeartDiseaseClassifier,
)
import dspy

# Load dataset
examples = load_heartdisease_dataset(split="train")
val_examples = examples[30:35]  # Just 5 examples
valset = create_dspy_examples(val_examples)

print(f"Loaded {len(valset)} validation examples")
print(f"First example expected: {valset[0].classification}")

# Configure LM
api_key = os.getenv("GROQ_API_KEY")
print(f"API key loaded: {api_key[:20]}...")

lm = dspy.LM("groq/openai/gpt-oss-20b", api_key=api_key, cache=False)
dspy.configure(lm=lm)

# Create module
module = HeartDiseaseClassifier()

# Test single prediction
print("\n=== Testing single prediction ===")
result = module(valset[0].features)
print(f"Prediction: {result.classification}")
print(f"Expected: {valset[0].classification}")
print(f"Score: {heartdisease_metric(valset[0], result)}")

# Test evaluation
print("\n=== Testing full evaluation ===")
from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=valset, metric=heartdisease_metric, num_threads=1, display_progress=True)
score = evaluate(module)

print(f"\nEvaluation result type: {type(score)}")
print(f"Evaluation result: {score}")
if isinstance(score, dict):
    print(f"Evaluation dict keys: {score.keys()}")
