"""Loaders for system specifications from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from synth_ai.sdk.specs.dataclasses import (
    Constraints,
    Example,
    GlossaryItem,
    Interfaces,
    Metadata,
    Principle,
    Rule,
    Spec,
    TestCase,
)


def load_spec_from_dict(data: Dict[str, Any]) -> Spec:
    """Load a Spec from a dictionary.
    
    Args:
        data: Dictionary representation of a spec
        
    Returns:
        Spec instance
    """
    # Load metadata
    metadata_dict = data["metadata"]
    md = Metadata(
        id=metadata_dict["id"],
        title=metadata_dict["title"],
        version=metadata_dict["version"],
        owner=metadata_dict.get("owner"),
        created_at=metadata_dict.get("created_at"),
        updated_at=metadata_dict.get("updated_at"),
        imports=metadata_dict.get("imports", []),
        scope=metadata_dict.get("scope"),
        description=metadata_dict.get("description"),
    )
    
    # Load principles
    principles = [
        Principle(
            id=p["id"],
            text=p["text"],
            rationale=p.get("rationale"),
        )
        for p in data.get("principles", [])
    ]
    
    # Load rules
    def load_rule(r: Dict[str, Any]) -> Rule:
        constraints_data = r.get("constraints", {})
        constraints = Constraints(**constraints_data)
        
        examples_data = r.get("examples", [])
        examples = [
            Example(
                kind=e["kind"],
                prompt=e["prompt"],
                response=e["response"],
                description=e.get("description"),
            )
            for e in examples_data
        ]
        
        tests_data = r.get("tests", [])
        tests = [
            TestCase(
                id=t["id"],
                challenge=t["challenge"],
                asserts=t.get("asserts", []),
                expected_behavior=t.get("expected_behavior"),
            )
            for t in tests_data
        ]
        
        return Rule(
            id=r["id"],
            title=r["title"],
            rationale=r.get("rationale"),
            constraints=constraints,
            examples=examples,
            tests=tests,
            priority=r.get("priority"),
        )
    
    rules = [load_rule(r) for r in data.get("rules", [])]
    
    # Load interfaces
    interfaces_data = data.get("interfaces", {})
    interfaces = Interfaces(**interfaces_data)
    
    # Load glossary
    glossary = [
        GlossaryItem(
            term=g["term"],
            definition=g["definition"],
            aliases=g.get("aliases", []),
        )
        for g in data.get("glossary", [])
    ]
    
    # Load changelog
    changelog = data.get("changelog", [])
    
    return Spec(
        metadata=md,
        principles=principles,
        rules=rules,
        interfaces=interfaces,
        glossary=glossary,
        changelog=changelog,
    )


def load_spec_from_file(path: str | Path) -> Spec:
    """Load a Spec from a JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Spec instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    return load_spec_from_dict(data)


