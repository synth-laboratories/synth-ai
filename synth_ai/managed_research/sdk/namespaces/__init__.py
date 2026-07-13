"""Namespace authorities for the Managed Research SDK."""

from synth_ai.managed_research.sdk.approvals import ApprovalsAPI
from synth_ai.managed_research.sdk.cost import RunCostAPI
from synth_ai.managed_research.sdk.credentials import CredentialsAPI
from synth_ai.managed_research.sdk.datasets import DatasetsAPI
from synth_ai.managed_research.sdk.environments import EnvironmentsAPI
from synth_ai.managed_research.sdk.exports import ExportsAPI
from synth_ai.managed_research.sdk.files import FilesAPI
from synth_ai.managed_research.sdk.github import GithubAPI
from synth_ai.managed_research.sdk.integrations import IntegrationsAPI
from synth_ai.managed_research.sdk.logs import LogsAPI
from synth_ai.managed_research.sdk.models import ModelsAPI
from synth_ai.managed_research.sdk.outputs import OutputsAPI
from synth_ai.managed_research.sdk.progress import ProgressAPI
from synth_ai.managed_research.sdk.projects import ProjectsAPI
from synth_ai.managed_research.sdk.prs import PrsAPI
from synth_ai.managed_research.sdk.readiness import ReadinessAPI
from synth_ai.managed_research.sdk.repos import ReposAPI
from synth_ai.managed_research.sdk.repositories import RepositoriesAPI
from synth_ai.managed_research.sdk.runs import RunsAPI
from synth_ai.managed_research.sdk.secrets import SecretsAPI
from synth_ai.managed_research.sdk.setup import SetupAPI
from synth_ai.managed_research.sdk.trained_models import TrainedModelsAPI
from synth_ai.managed_research.sdk.usage import UsageAPI
from synth_ai.managed_research.sdk.workspace_inputs import WorkspaceInputsAPI

__all__ = [
    "ApprovalsAPI",
    "CredentialsAPI",
    "DatasetsAPI",
    "EnvironmentsAPI",
    "ExportsAPI",
    "FilesAPI",
    "GithubAPI",
    "IntegrationsAPI",
    "LogsAPI",
    "ModelsAPI",
    "OutputsAPI",
    "ProgressAPI",
    "ProjectsAPI",
    "PrsAPI",
    "ReadinessAPI",
    "RepositoriesAPI",
    "ReposAPI",
    "RunCostAPI",
    "RunsAPI",
    "SecretsAPI",
    "SetupAPI",
    "TrainedModelsAPI",
    "UsageAPI",
    "WorkspaceInputsAPI",
]
