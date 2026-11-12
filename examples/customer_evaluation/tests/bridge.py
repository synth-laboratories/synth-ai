"""Task-owned Rhodes adapter; candidate bytes never enter the verifier image."""

import json
import sys

from verify import score

session = json.loads(sys.stdin.readline())
assert session["schema_version"] == "synth.evaluator-session.v1"
inputs = session["instance"]["inputs"]
print(json.dumps({"kind": "invoke", "id": "sort", "input": inputs}), flush=True)
response = json.loads(sys.stdin.readline())
assert response["kind"] == "response" and response["id"] == "sort"
print(
    json.dumps(
        {
            "kind": "result",
            "result": {
                "schema_version": "synth.evaluation-result.v1",
                "score": score(inputs, response["output"]),
                "metrics": {"items": float(len(inputs["values"]))},
            },
        }
    ),
    flush=True,
)
