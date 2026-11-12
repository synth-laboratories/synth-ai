"""A customer policy, installed only in the separate candidate container."""

import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"values": sorted(set(request["values"]))}), flush=True)
