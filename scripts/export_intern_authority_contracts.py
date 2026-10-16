"""Generate producer-owned schemas for the curated Intern SDK authority APIs."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=Path, required=True)
args = parser.parse_args()
backend = args.backend.resolve()
if not backend.is_relative_to(Path("/Users/joshuapurtell/GitHub")):
    raise ValueError("backend producer must remain in shared workspace")
sys.path.insert(0, str(backend))
identities = importlib.import_module("packages.intern.identity_api")
routes = importlib.import_module("packages.intern.runtime_route_api")
grants = importlib.import_module("packages.intern.task_grants")
models = {
    "InternIdentityProvisionRequest": identities.InternIdentityProvisionRequest,
    "InternIdentity": identities.InternIdentityResponse,
    "InternIdentityProvisionReceipt": identities.InternIdentityProvisionResponse,
    "InternIdentitySelection": identities.InternIdentitySelectionResponse,
    "InternIdentityCatalogPage": identities.InternIdentityCatalogResponse,
    "InternRuntimeRouteSelectionRequest": routes.InternRuntimeRouteSelectionRequest,
    "InternRuntimeRouteView": routes.InternRuntimeRouteResponse,
    "InternRuntimeRouteReceipt": routes.InternRuntimeRouteReceipt,
    "InternTaskGrantDeclaration": grants.InternTaskGrantDeclaration,
    "BackendContextBindGrantDeclaration": grants.BackendContextBindGrantDeclaration,
    "InternTaskGrantRevocation": grants.InternTaskGrantRevocation,
}
source_paths = [
    "packages/intern/identity_api.py",
    "packages/intern/runtime_route_api.py",
    "packages/intern/task_grants.py",
]
document = {
    "schema_version": "sdk.intern_authority_producer.v1",
    "sources": {
        name: hashlib.sha256((backend / name).read_bytes()).hexdigest() for name in source_paths
    },
    "task_operations": sorted(grants.TASK_OPERATIONS),
    "models": {name: model.model_json_schema() for name, model in models.items()},
}
root = Path(__file__).resolve().parents[1]
(root / "synth_ai/sdk/research/contracts/intern_authority_producer.json").write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n"
)
print("Generated 11 owning Intern schemas with source hashes and closed Task vocabulary.")
