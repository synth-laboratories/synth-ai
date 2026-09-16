from synth_ai.sdk.index.catalog import Capabilities
from synth_ai.sdk.index.search import SearchMode


def test_capabilities_accept_joined_fast_and_deep_runtime():
    capabilities = Capabilities.model_validate(
        {
            "contribution_schema_versions": ["synth.contribution.v1"],
            "taxonomy_version": "synth.index.taxonomy.v1",
            "modes": ["fast", "deep"],
            "search_modes": ["fast", "deep"],
            "visibilities": ["public", "private"],
            "deep_search": True,
            "search_filters": False,
            "private_search": {"activated": False},
            "upload": {"enabled": True},
            "review": {"enabled": True},
            "publication": {"enabled": True},
            "limits": {
                "max_results": 10,
                "max_excerpts_per_result": 2,
                "query_max_bytes": 8192,
                "contents_max_bytes": 65536,
            },
        }
    )
    assert capabilities.modes == (SearchMode.FAST, SearchMode.DEEP)
    assert capabilities.search_modes == (SearchMode.FAST, SearchMode.DEEP)
    assert capabilities.deep_search is True
