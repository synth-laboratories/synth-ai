import synth_ai.managed_research
import synth_ai.managed_research.sdk as managed_research_sdk
import synth_ai.managed_research.transport as transport
from synth_ai import ManagedResearchClient, ManagedResearchMcpServer, ResearchClient, SynthClient
from synth_ai.managed_research import SmrControlClient


def test_synth_ai_top_level_exports_research_surface() -> None:
    assert synth_ai.ManagedResearchClient is ManagedResearchClient
    assert synth_ai.ManagedResearchMcpServer is ManagedResearchMcpServer
    assert synth_ai.SmrControlClient is SmrControlClient

    client = SynthClient(api_key="test-key", base_url="http://localhost:8000")
    assert isinstance(client.research, ResearchClient)
    assert client.research is client.research


def test_top_level_public_exports_match_rewritten_surface() -> None:
    assert synth_ai.managed_research.__version__
    assert synth_ai.managed_research.SmrControlClient is SmrControlClient
    assert {
        "CredentialsAPI",
        "DatasetsAPI",
        "ExportsAPI",
        "FilesAPI",
        "GithubAPI",
        "ModelsAPI",
        "OPENAI_TRANSPORT_MODE_AUTO",
        "OPENAI_TRANSPORT_MODE_BACKEND_BFF",
        "OPENAI_TRANSPORT_MODE_DIRECT_HP",
        "OutputsAPI",
        "ProjectReadiness",
        "ProgressAPI",
        "PrsAPI",
        "ReadinessAPI",
        "ReposAPI",
        "RunProgress",
        "SmrBranchMode",
        "SmrControlClient",
        "SmrLogicalTimeline",
        "SmrRunBranchRequest",
        "SmrRunBranchResponse",
        "WorkspaceInputsAPI",
        "WorkspaceInputsState",
        "WorkspaceUploadResult",
    }.issubset(set(synth_ai.managed_research.__all__))


def test_sdk_exports_cover_new_namespaces() -> None:
    assert {
        "CredentialsAPI",
        "DatasetsAPI",
        "ExportsAPI",
        "FilesAPI",
        "GithubAPI",
        "ManagedResearchProjectClient",
        "ModelsAPI",
        "OPENAI_TRANSPORT_MODE_AUTO",
        "OPENAI_TRANSPORT_MODE_BACKEND_BFF",
        "OPENAI_TRANSPORT_MODE_DIRECT_HP",
        "OutputsAPI",
        "ProgressAPI",
        "PrsAPI",
        "ProjectsAPI",
        "ReadinessAPI",
        "ReposAPI",
        "RunsAPI",
        "SmrBranchMode",
        "SmrControlClient",
        "SmrLogicalTimeline",
        "SmrRunBranchRequest",
        "SmrRunBranchResponse",
        "WorkspaceInputsAPI",
    }.issubset(set(managed_research_sdk.__all__))


def test_transport_exports_cover_helper_surface() -> None:
    assert {
        "BinaryPayloadPreview",
        "RetryPolicy",
        "SmrHttpTransport",
        "build_query_params",
        "extract_next_cursor",
        "preview_binary_payload",
    }.issubset(set(transport.__all__))
