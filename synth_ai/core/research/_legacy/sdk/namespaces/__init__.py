"""Namespace authorities for the Managed Research SDK."""

from synth_ai.core.research._legacy.sdk.approvals import ApprovalsAPI
from synth_ai.core.research._legacy.sdk.cost import RunCostAPI
from synth_ai.core.research._legacy.sdk.credentials import CredentialsAPI
from synth_ai.core.research._legacy.sdk.datasets import DatasetsAPI
from synth_ai.core.research._legacy.sdk.environments import EnvironmentsAPI
from synth_ai.core.research._legacy.sdk.exports import ExportsAPI
from synth_ai.core.research._legacy.sdk.files import FilesAPI
from synth_ai.core.research._legacy.sdk.github import GithubAPI
from synth_ai.core.research._legacy.sdk.integrations import IntegrationsAPI
from synth_ai.core.research._legacy.sdk.logs import LogsAPI
from synth_ai.core.research._legacy.sdk.models import ModelsAPI
from synth_ai.core.research._legacy.sdk.outputs import OutputsAPI
from synth_ai.core.research._legacy.sdk.progress import ProgressAPI
from synth_ai.core.research._legacy.sdk.projects import ProjectsAPI
from synth_ai.core.research._legacy.sdk.prs import PrsAPI
from synth_ai.core.research._legacy.sdk.readiness import ReadinessAPI
from synth_ai.core.research._legacy.sdk.repos import ReposAPI
from synth_ai.core.research._legacy.sdk.repositories import RepositoriesAPI
from synth_ai.core.research._legacy.sdk.runs import RunsAPI
from synth_ai.core.research._legacy.sdk.secrets import SecretsAPI
from synth_ai.core.research._legacy.sdk.setup import SetupAPI
from synth_ai.core.research._legacy.sdk.trained_models import TrainedModelsAPI
from synth_ai.core.research._legacy.sdk.usage import UsageAPI
from synth_ai.core.research._legacy.sdk.workspace_inputs import WorkspaceInputsAPI

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
