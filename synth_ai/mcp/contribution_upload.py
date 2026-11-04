"""Bounded exact private archives; no paths or transfer grants cross hosted MCP.

See SDK_0230_MCP.md. Prepare/status/finalize keep separate native identities.
Archive bytes belong in programmatic calls, not model-generated text.
"""

import base64
import binascii
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from synth_ai.options import MutationOptions, TransferOptions
from synth_ai.types.base import Identifier, Model, ResponseIdentifier, ResponseModel
from synth_ai.types.uploads import ContributionUploadManifest, UploadSource

from .contracts import ContributionIdentity

MAXIMUM_ARCHIVE_BYTES = 4 * 1024 * 1024
MAXIMUM_ENCODED_BYTES = 4 * ((MAXIMUM_ARCHIVE_BYTES + 2) // 3)


class UploadPrepare(ContributionUploadManifest, ContributionIdentity):
    idempotency_key: Identifier


class UploadRead(Model):
    upload_id: Identifier


class UploadFinalize(UploadRead):
    idempotency_key: Identifier
    archive_base64: str | None = Field(
        default=None, min_length=1, max_length=MAXIMUM_ENCODED_BYTES, repr=False
    )


class UploadProgress(ResponseModel):
    upload_id: ResponseIdentifier
    contribution_id: ResponseIdentifier
    state: Literal["prepared", "received", "finalized", "expired", "refused"]
    manifest: ContributionUploadManifest
    expires_at: datetime | None


def progress(status):
    return UploadProgress.model_validate_json(
        status.model_dump_json(exclude={"transfer_url", "transfer_headers"})
    )


def archive_bytes(value):
    try:
        content = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("archive must be canonical base64") from error
    if (
        not 1 <= len(content) <= MAXIMUM_ARCHIVE_BYTES
        or base64.b64encode(content).decode() != value
    ):
        raise ValueError("archive exceeds its bound or uses noncanonical base64")
    return content


async def prepare(client, args):
    manifest = ContributionUploadManifest.model_validate(
        args.model_dump(exclude={"contribution_id", "idempotency_key"})
    )
    session = await client.hidden("index").uploads.prepare(
        args.contribution_id,
        manifest=manifest,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )
    return progress(await session.status())


async def get(client, args):
    session = await client.hidden("index").uploads.get(args.upload_id)
    return progress(await session.status())


async def finalize(client, args):
    session = await client.hidden("index").uploads.get(args.upload_id)
    if args.archive_base64 is not None:
        content = archive_bytes(args.archive_base64)
        status = await session.status()
        if (
            len(content) != status.manifest.byte_size
            or hashlib.sha256(content).hexdigest() != status.manifest.digest_sha256
        ):
            raise ValueError("archive does not match prepared manifest")
        if status.state != "finalized":
            with tempfile.TemporaryDirectory(prefix="synth-mcp-contribution-") as directory:
                path = Path(directory) / "package.archive"
                path.write_bytes(content)
                source = UploadSource.from_path(
                    path, media_type=status.manifest.media_type, maximum_bytes=MAXIMUM_ARCHIVE_BYTES
                )
                if source.manifest != status.manifest:
                    raise ValueError("archive does not match prepared manifest")
                await session.transfer(
                    source,
                    transfer=TransferOptions(
                        timeout_seconds=90,
                        maximum_bytes=MAXIMUM_ARCHIVE_BYTES,
                        chunk_bytes=65536,
                        retry_count_max=0,
                    ),
                )
    return await session.finalize(mutation=MutationOptions(idempotency_key=args.idempotency_key))
