import pytest
from synth_ai.environments.examples.enron.art_helpers.email_search_tools import search_emails


@pytest.mark.parametrize("kw", [["enron"]])  # , ["meeting"], ["energy"]
def test_index_has_hits(kw):
    hits = search_emails(inbox="john.lavorato@enron.com", keywords=kw)
    assert len(hits) > 0, f"no hits for {kw}"
