"""Original fixed-seed synthetic metric audit; standard library, no network."""
import json
import random
from pathlib import Path
rng = random.Random(20260930)
labels = [0] * 90 + [1] * 10
rng.shuffle(labels)
rows = [{"row": i, "label": label, "feature": label} for i, label in enumerate(labels)]
predictors = {"majority": [0] * len(rows), "feature_rule": [r["feature"] for r in rows]}
results = {}
for name, predictions in predictors.items():
    per_class = {str(c): sum(p == y for p, y in zip(predictions, labels) if y == c) / labels.count(c) for c in (0, 1)}
    results[name] = {"accuracy": sum(p == y for p, y in zip(predictions, labels)) / len(labels), "per_class_accuracy": per_class, "macro_accuracy": sum(per_class.values()) / 2}
Path('inputs.json').write_text(json.dumps(rows, sort_keys=True, indent=2) + '\n')
Path('outputs.json').write_text(json.dumps({"seed": 20260930, "predictions": predictors, "metrics": results}, sort_keys=True, indent=2) + '\n')
assert results['majority']['accuracy'] == 0.9
assert results['majority']['macro_accuracy'] == 0.5
assert results['feature_rule']['accuracy'] == 1.0
print(json.dumps(results, sort_keys=True))
