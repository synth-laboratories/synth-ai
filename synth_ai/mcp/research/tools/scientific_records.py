"""Scientific MCP schemas from producer-owned contracts. See SYN-3559."""

import base64

from pydantic import Field
from synth_ai.mcp.research.registry import READ_SCOPES, WRITE_SCOPES, ToolDefinition
from synth_ai.sdk.research.contracts.forge.contracts import Contract, Identifier
from synth_ai.sdk.research.contracts.forge.records import Artifact, Record


class WriteArguments(Contract):
    project_id: Identifier
    record_id: Identifier
    operation_id: Identifier
    expected_revision: int = Field(default=0, ge=0)
    payload: Record


class ReadArguments(Contract):
    project_id: Identifier
    record_id: Identifier
    revision: int | None = Field(default=None, ge=1)


class ArtifactReadArguments(Contract):
    project_id: Identifier
    record_id: Identifier
    revision: int = Field(ge=1)


class ReceiptArguments(Contract):
    project_id: Identifier
    operation_id: Identifier


class ListArguments(Contract):
    project_id: Identifier
    kind: Identifier
    after: str = ""
    limit: int = Field(default=100, ge=1, le=1000)


class EventArguments(Contract):
    project_id: Identifier
    after: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)


READ_DESCRIPTIONS = {
    "research_verify_citations": (
        "Check a retained record revision's external citations: retained, dangling, "
        "denied, conflict, unavailable or refused. Read-only."
    ),
}


def build_scientific_record_tools(client_factory):
    tools = []
    for name, kind in (
        ("research_save_record", None),
        ("research_create_log", "research_log"),
        ("research_append_log_entry", "log_entry"),
        ("research_save_report", "report"),
        ("research_attach_artifact", "artifact"),
        ("research_create_experiment_revision", "experiment"),
        ("research_register_trial", "trial"),
        ("research_record_result", "result"),
        ("research_review_revision", "review"),
    ):

        def writer(expected_kind):
            def handler(arguments):
                selected = WriteArguments.model_validate(arguments)
                if expected_kind is not None and selected.payload.kind != expected_kind:
                    raise ValueError("payload kind differs from selected scientific operation")
                with client_factory({}) as client:
                    return client.records.write(
                        selected.project_id,
                        selected.payload,
                        record_id=selected.record_id,
                        operation_id=selected.operation_id,
                        expected_revision=selected.expected_revision,
                    ).model_dump(mode="json")

            return handler

        schema = WriteArguments.model_json_schema()
        if kind is not None:
            # Preserve all producer $defs, narrowing only the selected operation.
            choices = schema["properties"]["payload"]["oneOf"]
            selected_choice = next(
                choice
                for choice in choices
                if schema["$defs"][choice["$ref"].split("/")[-1]]["properties"]["kind"].get("const")
                == kind
            )
            schema["properties"]["payload"] = selected_choice
        tools.append(
            ToolDefinition(
                name=name,
                description=f"Save an immutable {kind or 'scientific record'} revision.",
                input_schema=schema,
                handler=writer(kind),
                required_scopes=WRITE_SCOPES,
            )
        )
    for name, model, method in (
        ("research_get_record", ReadArguments, "get"),
        ("research_verify_citations", ReadArguments, "verify_citations"),
        ("research_list_records", ListArguments, "list"),
        ("research_get_operation_receipt", ReceiptArguments, "receipt"),
        ("research_record_events", EventArguments, "events"),
    ):

        def reader(argument_model, selected_method):
            def handler(arguments):
                selected = argument_model.model_validate(arguments)
                values = selected.model_dump()
                project_id = values.pop("project_id")
                with client_factory({}) as client:
                    result = getattr(client.records, selected_method)(project_id, **values)
                if isinstance(result, tuple):
                    return [row.model_dump(mode="json") for row in result]
                return result.model_dump(mode="json")

            return handler

        tools.append(
            ToolDefinition(
                name=name,
                description=READ_DESCRIPTIONS.get(
                    name, "Read scoped retained scientific evidence."
                ),
                input_schema=model.model_json_schema(),
                handler=reader(model, method),
                required_scopes=READ_SCOPES,
            )
        )

    def read_artifact(arguments):
        selected = ArtifactReadArguments.model_validate(arguments)
        with client_factory({}) as client:
            record = client.records.get(
                selected.project_id, selected.record_id, revision=selected.revision
            )
            if (
                not isinstance(record.payload, Artifact)
                or record.payload.byte_count > 4 * 1024 * 1024
            ):
                raise ValueError("MCP artifact delivery requires an artifact up to 4 MiB")
            content = client.records.download_artifact(
                selected.project_id, selected.record_id, revision=selected.revision
            )
        return {
            "reference": record.reference.model_dump(mode="json"),
            "encoding": "base64",
            "content": base64.b64encode(content).decode("ascii"),
        }

    tools.append(
        ToolDefinition(
            name="research_read_artifact",
            description="Read exact authorized artifact bytes up to 4 MiB.",
            input_schema=ArtifactReadArguments.model_json_schema(),
            handler=read_artifact,
            required_scopes=READ_SCOPES,
        )
    )
    return tools
