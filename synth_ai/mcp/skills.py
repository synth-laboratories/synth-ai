"""Packaged Cloud Swarms Agent Skills over MCP (SEP-2640 / io.modelcontextprotocol/skills)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from pydantic import JsonValue

__all__ = [
    "EXTENSION_CAPABILITIES",
    "EXTENSION_METHODS",
    "EXTENSION_URI",
    "INVALID_PARAMS",
    "RESOURCE_NOT_FOUND",
    "SkillCatalog",
    "SkillDirEntry",
    "SkillEntry",
    "SkillError",
    "SkillResource",
    "dispatch_extension_request",
    "get_skill_payload",
    "list_skills_payload",
    "load_skills",
    "parse_skill_uri",
    "read_directory_payload",
    "read_resource_entry",
]

SKILL_SCHEME = "skill://"
EXTENSION_URI = "io.modelcontextprotocol/skills"
EXTENSION_CAPABILITIES: dict[str, dict[str, JsonValue]] = {
    EXTENSION_URI: {"directoryRead": True},
}
EXTENSION_METHODS: frozenset[str] = frozenset(
    {"skills/list", "skills/get", "resources/directory/read"}
)
INVALID_PARAMS = -32602
RESOURCE_NOT_FOUND = -32002
MAX_SKILL_URI_LEN = 256

_MIME_BY_SUFFIX: dict[str, str] = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


class SkillError(ValueError):
    """Fail-closed JSON-RPC error for SEP-2640 skill and resource lookups."""

    def __init__(
        self,
        code: int,
        message: str,
        *,
        data: dict[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True, slots=True)
class SkillResource:
    uri: str
    name: str
    description: str
    mime_type: str
    size: int
    digest: str
    content: str


@dataclass(frozen=True, slots=True)
class SkillEntry:
    uri: str
    name: str
    description: str
    frontmatter: tuple[tuple[str, str], ...]
    revision: str
    resources: tuple[SkillResource, ...]


@dataclass(frozen=True, slots=True)
class SkillDirEntry:
    uri: str
    name: str
    is_directory: bool
    size: int | None = None
    digest: str | None = None


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    skills: tuple[SkillEntry, ...]
    by_skill_uri: dict[str, SkillEntry]
    by_resource_uri: dict[str, SkillResource]
    resources: tuple[SkillResource, ...]
    dirs: dict[str, tuple[SkillDirEntry, ...]]


def _valid_segment(segment: str) -> bool:
    if not segment or segment in {".", ".."} or len(segment) > 128:
        return False
    first = segment[0]
    if not (first.isascii() and first.isalnum()):
        return False
    return all(ch.isascii() and (ch.isalnum() or ch in "._-") for ch in segment)


def _valid_frontmatter_key(key: str) -> bool:
    if not key:
        return False
    return all(ch.isascii() and (ch.islower() or ch.isdigit() or ch in "_-") for ch in key)


def parse_skill_uri(raw: str, *, want_dir: bool = False) -> str | None:
    """Validate a skill:// URI fail-closed against traversal, injection, and scheme confusion."""
    if not raw or len(raw) > MAX_SKILL_URI_LEN or not raw.startswith(SKILL_SCHEME):
        return None
    if any(ch in raw for ch in "?#\\%"):
        return None
    if any(ord(ch) < 0x20 or ord(ch) >= 0x7F for ch in raw):
        return None
    rest = raw[len(SKILL_SCHEME) :]
    if want_dir:
        if not rest.endswith("/"):
            return None
        trimmed = rest[:-1]
        if not trimmed:
            return None
        parts = trimmed.split("/")
        if not all(_valid_segment(seg) for seg in parts):
            return None
        return SKILL_SCHEME + "/".join(parts) + "/"
    if rest.endswith("/"):
        return None
    parts = rest.split("/")
    if len(parts) < 2 or not all(_valid_segment(seg) for seg in parts):
        return None
    return SKILL_SCHEME + "/".join(parts)


def _parse_frontmatter(content: str) -> dict[str, str]:
    normalized = content.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise ValueError("missing opening frontmatter fence")
    rest = normalized[len("---\n") :]
    end = rest.find("\n---\n")
    if end < 0:
        if rest.endswith("\n---"):
            end = len(rest) - len("\n---")
        else:
            raise ValueError("missing closing frontmatter fence")
    block = rest[:end]
    parsed: dict[str, str] = {}
    for raw_line in block.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"malformed frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not _valid_frontmatter_key(key):
            raise ValueError(f"invalid frontmatter key: {key!r}")
        if key in parsed:
            raise ValueError(f"duplicate frontmatter key: {key!r}")
        if len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]
        parsed[key] = value
    return parsed


def _mime_type_for(name: str) -> str:
    dot = name.rfind(".")
    suffix = name[dot:].lower() if dot >= 0 else ""
    mime = _MIME_BY_SUFFIX.get(suffix)
    if mime is None:
        raise ValueError(f"unsupported skill resource extension: {name!r}")
    return mime


def _walk_skill_dir(
    node: Traversable,
    skill_name: str,
    rel_parts: tuple[str, ...],
    resources: list[SkillResource],
    dir_children: dict[str, dict[str, SkillDirEntry]],
) -> None:
    current_dir_uri = (
        f"{SKILL_SCHEME}{skill_name}/"
        if not rel_parts
        else f"{SKILL_SCHEME}{skill_name}/{'/'.join(rel_parts)}/"
    )
    dir_children.setdefault(current_dir_uri, {})
    for child in sorted(node.iterdir(), key=lambda item: item.name):
        name = child.name
        if not _valid_segment(name):
            raise ValueError(f"invalid skill path segment: {name!r}")
        next_parts = (*rel_parts, name)
        rel = "/".join(next_parts)
        if child.is_dir():
            sub_uri = f"{SKILL_SCHEME}{skill_name}/{rel}/"
            dir_children[current_dir_uri][name] = SkillDirEntry(
                uri=sub_uri, name=name, is_directory=True
            )
            _walk_skill_dir(child, skill_name, next_parts, resources, dir_children)
        elif child.is_file():
            raw = child.read_bytes()
            text = raw.decode("utf-8")
            mime = _mime_type_for(name)
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            uri = f"{SKILL_SCHEME}{skill_name}/{rel}"
            description = f"Reference resource for skill {skill_name}"
            resources.append(
                SkillResource(
                    uri=uri,
                    name=f"{skill_name}/{rel}",
                    description=description,
                    mime_type=mime,
                    size=len(raw),
                    digest=digest,
                    content=text,
                )
            )
            dir_children[current_dir_uri][name] = SkillDirEntry(
                uri=uri,
                name=name,
                is_directory=False,
                size=len(raw),
                digest=digest,
            )


def _load_single_skill(
    skill_dir: Traversable, skill_name: str
) -> tuple[SkillEntry, dict[str, tuple[SkillDirEntry, ...]]]:
    raw_resources: list[SkillResource] = []
    dir_children: dict[str, dict[str, SkillDirEntry]] = {}
    _walk_skill_dir(skill_dir, skill_name, (), raw_resources, dir_children)

    skill_uri = f"{SKILL_SCHEME}{skill_name}/SKILL.md"
    skill_doc = next((item for item in raw_resources if item.uri == skill_uri), None)
    if skill_doc is None:
        raise ValueError(f"skill {skill_name!r} is missing SKILL.md")
    frontmatter = _parse_frontmatter(skill_doc.content)
    if frontmatter.get("name") != skill_name:
        raise ValueError(
            f"skill {skill_name!r} frontmatter name {frontmatter.get('name')!r} mismatches directory"
        )
    description = frontmatter.get("description", "")
    if not description:
        raise ValueError(f"skill {skill_name!r} frontmatter description is required")

    resources = tuple(
        sorted(
            (
                SkillResource(
                    uri=item.uri,
                    name=item.name,
                    description=description if item.uri == skill_uri else item.description,
                    mime_type=item.mime_type,
                    size=item.size,
                    digest=item.digest,
                    content=item.content,
                )
                for item in raw_resources
            ),
            key=lambda item: item.uri,
        )
    )
    rev_hash = hashlib.sha256()
    for item in resources:
        rev_hash.update(item.uri.encode("utf-8"))
        rev_hash.update(b"\x00")
        rev_hash.update(item.digest.encode("utf-8"))
        rev_hash.update(b"\n")
    revision = "sha256:" + rev_hash.hexdigest()

    dirs: dict[str, tuple[SkillDirEntry, ...]] = {
        dir_uri: tuple(sorted(children.values(), key=lambda item: item.name))
        for dir_uri, children in dir_children.items()
    }
    entry = SkillEntry(
        uri=skill_uri,
        name=skill_name,
        description=description,
        frontmatter=tuple(sorted(frontmatter.items())),
        revision=revision,
        resources=resources,
    )
    return entry, dirs


@lru_cache(maxsize=1)
def load_skills() -> SkillCatalog:
    """Load and validate the packaged skill bundle once with deterministic digests."""
    root = files("synth_ai.mcp").joinpath("skills")
    if not root.is_dir():
        raise ValueError("packaged MCP skills directory is missing")
    entries: list[SkillEntry] = []
    by_skill_uri: dict[str, SkillEntry] = {}
    by_resource_uri: dict[str, SkillResource] = {}
    all_resources: list[SkillResource] = []
    all_dirs: dict[str, tuple[SkillDirEntry, ...]] = {}

    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            raise ValueError(f"unexpected top-level file in MCP skills bundle: {child.name!r}")
        if not _valid_segment(child.name):
            raise ValueError(f"invalid skill directory name: {child.name!r}")
        entry, skill_dirs = _load_single_skill(child, child.name)
        entries.append(entry)
        by_skill_uri[entry.uri] = entry
        for resource in entry.resources:
            by_resource_uri[resource.uri] = resource
            all_resources.append(resource)
        all_dirs.update(skill_dirs)

    if not entries:
        raise ValueError("packaged MCP skills bundle is empty")
    entries.sort(key=lambda item: item.name)
    all_resources.sort(key=lambda item: item.uri)
    return SkillCatalog(
        skills=tuple(entries),
        by_skill_uri=by_skill_uri,
        by_resource_uri=by_resource_uri,
        resources=tuple(all_resources),
        dirs=all_dirs,
    )


def _skill_payload(entry: SkillEntry) -> dict[str, JsonValue]:
    frontmatter: dict[str, JsonValue] = dict(entry.frontmatter)
    resources: list[JsonValue] = [
        {
            "uri": item.uri,
            "mimeType": item.mime_type,
            "size": item.size,
            "digest": item.digest,
        }
        for item in entry.resources
    ]
    return {
        "uri": entry.uri,
        "name": entry.name,
        "description": entry.description,
        "frontmatter": frontmatter,
        "revision": entry.revision,
        "resources": resources,
    }


def list_skills_payload() -> dict[str, JsonValue]:
    catalog = load_skills()
    items: list[JsonValue] = [_skill_payload(entry) for entry in catalog.skills]
    return {"skills": items}


def get_skill_payload(uri: str) -> dict[str, JsonValue]:
    normalized = parse_skill_uri(uri, want_dir=False)
    if normalized is None:
        raise SkillError(
            INVALID_PARAMS,
            "Invalid params: malformed or disallowed skill URI.",
            data={"uri": uri},
        )
    entry = load_skills().by_skill_uri.get(normalized)
    if entry is None:
        raise SkillError(RESOURCE_NOT_FOUND, "Skill not found.", data={"uri": normalized})
    return _skill_payload(entry)


def read_resource_entry(uri: str) -> SkillResource:
    normalized = parse_skill_uri(uri, want_dir=False)
    if normalized is None:
        raise SkillError(
            INVALID_PARAMS,
            "Invalid params: malformed or disallowed resource URI.",
            data={"uri": uri},
        )
    entry = load_skills().by_resource_uri.get(normalized)
    if entry is None:
        raise SkillError(RESOURCE_NOT_FOUND, "Resource not found.", data={"uri": normalized})
    return entry


def read_directory_payload(uri: str) -> dict[str, JsonValue]:
    normalized = parse_skill_uri(uri, want_dir=True)
    if normalized is None:
        raise SkillError(
            INVALID_PARAMS,
            "Invalid params: malformed or disallowed directory URI.",
            data={"uri": uri},
        )
    children = load_skills().dirs.get(normalized)
    if children is None:
        raise SkillError(
            RESOURCE_NOT_FOUND, "Resource directory not found.", data={"uri": normalized}
        )
    items: list[JsonValue] = []
    for child in children:
        entry: dict[str, JsonValue] = {
            "uri": child.uri,
            "name": child.name,
            "isDirectory": child.is_directory,
        }
        if not child.is_directory:
            entry["size"] = child.size
            entry["digest"] = child.digest
        items.append(entry)
    return {"entries": items}


def dispatch_extension_request(
    method: str, params: dict[str, JsonValue] | None
) -> tuple[dict[str, JsonValue] | None, SkillError | None]:
    """Handle SEP-2640 extension RPC methods fail-closed."""
    try:
        if method == "skills/list":
            return list_skills_payload(), None
        if method == "skills/get":
            uri = params.get("uri") if isinstance(params, dict) else None
            if not isinstance(uri, str):
                raise SkillError(INVALID_PARAMS, "Invalid params: uri is required.")
            return get_skill_payload(uri), None
        if method == "resources/directory/read":
            uri = params.get("uri") if isinstance(params, dict) else None
            if not isinstance(uri, str):
                raise SkillError(INVALID_PARAMS, "Invalid params: uri is required.")
            return read_directory_payload(uri), None
    except SkillError as error:
        return None, error
    return None, SkillError(-32601, f"Method '{method}' not found.")
