import json
import pytest

from synth_ai.sdk.learning.sft.data import parse_jsonl_line, SFTDataError


@pytest.mark.parametrize(
    "payload",
    [
        # message missing role
        {"messages": [{"content": "hi"}]},
        # tool_calls wrong type
        {"messages": [{"role": "user", "content": "hi", "tool_calls": {}}]},
        # content wrong type
        {"messages": [{"role": "user", "content": 123}]},
    ],
)
def test_parse_rejects_malformed_messages(payload):
    with pytest.raises(SFTDataError):
        parse_jsonl_line(json.dumps(payload))


