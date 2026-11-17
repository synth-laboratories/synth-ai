"""Map validated MCP input to public SDK calls. See docs/protocols.md.

The JSON mapping exists only at this serialization boundary. Each operation
parses its contract body before invoking a typed SDK method.
"""

# ruff: noqa: UP045 -- Public boundary optionals use Synth Style's Optional vocabulary.
from collections.abc import Mapping
from typing import Optional

from pydantic import BaseModel, JsonValue, TypeAdapter

from .._generated.models import (
    CreateEvaluation,
    EvaluationInput,
    LaunchRequest,
    RunLabelsRequest,
    RunLimitUpdate,
    UploadFileRequest,
)
from .._limits import SERVER_WAIT_SECONDS_DEFAULT
from .._requests import DomicileRequirement
from ..aclient import AsyncSynthClient
from ..evaluations import EvaluationInputKind
from ._arguments import optional, page_limit, required
from .resource_dispatch import dispatch_resource

__all__ = ["dispatch"]


async def dispatch(
    client: AsyncSynthClient, operation: str, arguments: Mapping[str, JsonValue]
) -> BaseModel | None:
    match operation:
        case "getAccount":
            return await client.account.get()
        case "getBalance":
            return await client.billing.balance()
        case "listRuns":
            return await client.swarms.list(
                limit=page_limit(arguments),
                cursor=optional(arguments, "cursor", str),
                state=optional(arguments, "state", str),
                created_after=optional(arguments, "created_after", str),
            )
        case "createRun":
            return await client.swarms.launch_request(
                required(arguments, "body", LaunchRequest),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getRun":
            return await client.swarms.get(required(arguments, "run_id", str))
        case "waitRun":
            return await client.swarms.wait_update(
                required(arguments, "run_id", str),
                timeout_seconds=required(arguments, "timeout_seconds", int)
                if "timeout_seconds" in arguments
                else SERVER_WAIT_SECONDS_DEFAULT,
            )
        case "cancelRun":
            return await client.swarms.cancel(
                required(arguments, "run_id", str),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "pauseRun":
            return await client.swarms.pause(
                required(arguments, "run_id", str),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "resumeRun":
            return await client.swarms.resume(
                required(arguments, "run_id", str),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "listRunEvents":
            return await client.swarms.events(
                required(arguments, "run_id", str),
                cursor=optional(arguments, "cursor", str),
                limit=page_limit(arguments),
            )
        case "listRunTasks":
            return await client.swarms.tasks(
                required(arguments, "run_id", str),
                cursor=optional(arguments, "cursor", str),
                limit=page_limit(arguments),
            )
        case "getRunSandboxes":
            return await client.swarms.sandboxes(required(arguments, "run_id", str))
        case "getRunReport":
            return await client.swarms.report(required(arguments, "run_id", str))
        case "getRunAllocation":
            return await client.swarms.allocation(required(arguments, "run_id", str))
        case "listRunOutputs":
            return await client.swarms.outputs(
                required(arguments, "run_id", str),
                cursor=optional(arguments, "cursor", str),
                limit=page_limit(arguments),
            )
        case "getRunOutput":
            return await client.swarms.output(
                required(arguments, "run_id", str),
                required(arguments, "output_id", str),
            )
        case "getRunOutputContent":
            return await client.swarms.output_inline(
                required(arguments, "run_id", str),
                required(arguments, "output_id", str),
                revision_id=required(arguments, "revision_id", str),
                digest_sha256=required(arguments, "digest_sha256", str),
            )
        case "getRunResult":
            return await client.swarms.result(required(arguments, "run_id", str))
        case "getLaunchChoices":
            return await client.swarms.launch_choices(
                profile=optional(arguments, "profile", str),
                domicile_requirement=TypeAdapter(Optional[DomicileRequirement]).validate_python(
                    arguments.get("domicile_requirement"),
                    strict=True,
                ),
                parallel_workers=optional(arguments, "parallel_workers", int),
                wall_time_seconds=optional(arguments, "wall_time_seconds", int),
                budget=optional(arguments, "maximum_charge", str),
            )
        case "listFamilies":
            return await client.swarms.families()
        case "listProfiles":
            return await client.swarms.profiles(
                research_task=optional(arguments, "research_task", str),
            )
        case "listUsageRecords":
            return await client.account.usage_records(
                run_id=optional(arguments, "run_id", str),
                since=optional(arguments, "since", str),
                until=optional(arguments, "until", str),
                cursor=optional(arguments, "cursor", str),
                limit=page_limit(arguments),
            )
        case "uploadFile":
            upload = required(arguments, "body", UploadFileRequest)
            return await client.swarms.upload_text(
                upload.content,
                name=upload.name,
                media_type=upload.media_type,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "updateRunLabels":
            labels = required(arguments, "body", RunLabelsRequest)
            return await client.swarms.update_labels(
                required(arguments, "run_id", str),
                labels.labels,
                expected_revision=labels.expected_revision,
            )
        case "getRunLimits":
            return await client.swarms.limits(required(arguments, "run_id", str))
        case "updateRunLimits":
            return await client.swarms.update_limits_request(
                required(arguments, "run_id", str),
                required(arguments, "body", RunLimitUpdate),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "registerEvaluationInput":
            return await client.evaluations.register(
                TypeAdapter(EvaluationInputKind).validate_python(arguments["kind"], strict=True),
                TypeAdapter(EvaluationInput).validate_python(arguments["body"], strict=True),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "listEvaluationInputs":
            return await client.evaluations.list_inputs(
                TypeAdapter(EvaluationInputKind).validate_python(arguments["kind"], strict=True),
                name=optional(arguments, "name", str),
            )
        case "getEvaluationInput":
            return await client.evaluations.get_input(
                TypeAdapter(EvaluationInputKind).validate_python(arguments["kind"], strict=True),
                required(arguments, "revision_id", str),
            )
        case "preflightEvaluation":
            quote = required(arguments, "body", CreateEvaluation)
            return await client.evaluations.preflight(
                quote.request,
                maximum_charge_micros=quote.maximum_charge_micros,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "createEvaluation":
            evaluation = required(arguments, "body", CreateEvaluation)
            return await client.evaluations.create(
                evaluation.request,
                maximum_charge_micros=evaluation.maximum_charge_micros,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getEvaluation":
            return await client.evaluations.get(required(arguments, "evaluation_id", str))
        case "cancelEvaluation":
            return await client.evaluations.cancel(
                required(arguments, "evaluation_id", str),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "listEvaluationRollouts":
            # Each declared view has its own typed SDK return model.
            if arguments.get("view") == "summary":
                return await client.evaluations.rollout_summary(
                    required(arguments, "evaluation_id", str),
                    cursor=optional(arguments, "cursor", str),
                    instance_ids=TypeAdapter(Optional[list[str]]).validate_python(
                        arguments.get("instance_ids"),
                        strict=True,
                    ),
                )
            return await client.evaluations.rollouts(
                required(arguments, "evaluation_id", str),
                cursor=optional(arguments, "cursor", str),
                instance_ids=TypeAdapter(Optional[list[str]]).validate_python(
                    arguments.get("instance_ids"),
                    strict=True,
                ),
            )
        case _:
            return await dispatch_resource(client, operation, arguments)
