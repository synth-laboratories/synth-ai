"""Namespace authorities for the Managed Research SDK."""

from synth_ai.core.research.session.approvals import ApprovalsAPI
from synth_ai.core.research.session.cost import RunCostAPI
from synth_ai.core.research.session.credentials import CredentialsAPI
from synth_ai.core.research.session.datasets import DatasetsAPI
from synth_ai.core.research.session.environments import EnvironmentsAPI
from synth_ai.core.research.session.exports import ExportsAPI
from synth_ai.core.research.session.files import FilesAPI
from synth_ai.core.research.session.github import GithubAPI
from synth_ai.core.research.session.integrations import IntegrationsAPI
from synth_ai.core.research.session.logs import LogsAPI
from synth_ai.core.research.session.models import ModelsAPI
from synth_ai.core.research.session.outputs import OutputsAPI
from synth_ai.core.research.session.progress import ProgressAPI
from synth_ai.core.research.session.projects import ProjectsAPI
from synth_ai.core.research.session.prs import PrsAPI
from synth_ai.core.research.session.readiness import ReadinessAPI
from synth_ai.core.research.session.repos import ReposAPI
from synth_ai.core.research.session.repositories import RepositoriesAPI
from synth_ai.core.research.session.runs import RunsAPI
from synth_ai.core.research.session.secrets import SecretsAPI
from synth_ai.core.research.session.setup import SetupAPI
from synth_ai.core.research.session.trained_models import TrainedModelsAPI
from synth_ai.core.research.session.usage import UsageAPI
from synth_ai.core.research.session.workspace_inputs import WorkspaceInputsAPI

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
