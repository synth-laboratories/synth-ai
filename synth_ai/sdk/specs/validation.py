"""Validation utilities for system specifications."""

from __future__ import annotations

from typing import Any, Dict, List

from synth_ai.sdk.specs.dataclasses import Spec


class SpecValidationError(Exception):
    """Raised when spec validation fails."""
    pass


class SpecValidator:
    """Validator for system specifications."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, spec: Spec, strict: bool = False) -> bool:
        """Validate a spec and collect errors/warnings.
        
        Args:
            spec: The spec to validate
            strict: If True, warnings are treated as errors
            
        Returns:
            True if validation passes (no errors, or warnings if not strict)
            
        Raises:
            SpecValidationError: If validation fails with errors
        """
        self.errors = []
        self.warnings = []
        
        self._validate_metadata(spec)
        self._validate_principles(spec)
        self._validate_rules(spec)
        self._validate_glossary(spec)
        self._validate_consistency(spec)
        
        if strict and self.warnings:
            self.errors.extend(self.warnings)
            self.warnings = []
        
        if self.errors:
            error_msg = "\n".join([f"  - {err}" for err in self.errors])
            raise SpecValidationError(f"Spec validation failed:\n{error_msg}")
        
        return True
    
    def _validate_metadata(self, spec: Spec) -> None:
        """Validate metadata fields."""
        md = spec.metadata
        
        if not md.id:
            self.errors.append("Metadata: 'id' is required")
        elif not md.id.startswith("spec."):
            self.warnings.append("Metadata: 'id' should start with 'spec.' prefix")
        
        if not md.title:
            self.errors.append("Metadata: 'title' is required")
        
        if not md.version:
            self.errors.append("Metadata: 'version' is required")
        elif not self._is_valid_semver(md.version):
            self.warnings.append(f"Metadata: version '{md.version}' is not valid semver (X.Y.Z)")
        
        if not md.scope:
            self.warnings.append("Metadata: 'scope' should be specified")
    
    def _validate_principles(self, spec: Spec) -> None:
        """Validate principles."""
        seen_ids = set()
        
        for i, principle in enumerate(spec.principles):
            if not principle.id:
                self.errors.append(f"Principle {i}: 'id' is required")
            elif principle.id in seen_ids:
                self.errors.append(f"Principle {i}: duplicate id '{principle.id}'")
            else:
                seen_ids.add(principle.id)
            
            if not principle.text:
                self.errors.append(f"Principle {principle.id}: 'text' is required")
            
            if not principle.id.startswith("P-"):
                self.warnings.append(f"Principle {principle.id}: id should start with 'P-' prefix")
    
    def _validate_rules(self, spec: Spec) -> None:
        """Validate rules."""
        seen_ids = set()
        
        for i, rule in enumerate(spec.rules):
            if not rule.id:
                self.errors.append(f"Rule {i}: 'id' is required")
            elif rule.id in seen_ids:
                self.errors.append(f"Rule {i}: duplicate id '{rule.id}'")
            else:
                seen_ids.add(rule.id)
            
            if not rule.title:
                self.errors.append(f"Rule {rule.id}: 'title' is required")
            
            if not rule.id.startswith("R-"):
                self.warnings.append(f"Rule {rule.id}: id should start with 'R-' prefix")
            
            # Validate constraints
            if not rule.constraints.must and not rule.constraints.must_not:
                self.warnings.append(
                    f"Rule {rule.id}: no constraints defined (must/must_not are empty)"
                )
            
            # Validate examples
            for j, example in enumerate(rule.examples):
                if example.kind not in ("good", "bad"):
                    self.errors.append(
                        f"Rule {rule.id}, Example {j}: kind must be 'good' or 'bad', got '{example.kind}'"
                    )
                
                if not example.prompt or not example.response:
                    self.errors.append(
                        f"Rule {rule.id}, Example {j}: both 'prompt' and 'response' are required"
                    )
            
            # Validate priority
            if rule.priority is not None and (not isinstance(rule.priority, int) or rule.priority < 1 or rule.priority > 10):
                self.errors.append(
                    f"Rule {rule.id}: priority must be an integer between 1 and 10, got {rule.priority}"
                )
    
    def _validate_glossary(self, spec: Spec) -> None:
        """Validate glossary."""
        seen_terms = set()
        
        for item in spec.glossary:
            term_lower = item.term.lower()
            
            if term_lower in seen_terms:
                self.errors.append(f"Glossary: duplicate term '{item.term}'")
            else:
                seen_terms.add(term_lower)
            
            if not item.definition:
                self.errors.append(f"Glossary: term '{item.term}' missing definition")
            
            # Check for duplicate aliases
            for alias in item.aliases:
                alias_lower = alias.lower()
                if alias_lower in seen_terms:
                    self.warnings.append(
                        f"Glossary: alias '{alias}' for term '{item.term}' conflicts with existing term"
                    )
                seen_terms.add(alias_lower)
    
    def _validate_consistency(self, spec: Spec) -> None:
        """Validate cross-references and consistency."""
        # Check for orphaned imports
        if spec.metadata.imports:
            self.warnings.append(
                f"Metadata: imports specified but not validated ({len(spec.metadata.imports)} imports)"
            )
        
        # Warn if no rules or principles
        if not spec.rules and not spec.principles:
            self.warnings.append("Spec has no rules or principles defined")
        
        # Check for rules without examples
        rules_without_examples = [r.id for r in spec.rules if not r.examples]
        if rules_without_examples:
            self.warnings.append(
                f"Rules without examples: {', '.join(rules_without_examples)}"
            )
    
    @staticmethod
    def _is_valid_semver(version: str) -> bool:
        """Check if version follows semver format (X.Y.Z)."""
        parts = version.split(".")
        if len(parts) != 3:
            return False
        try:
            for part in parts:
                int(part)
            return True
        except ValueError:
            return False


def validate_spec_dict(data: Dict[str, Any], strict: bool = False) -> List[str]:
    """Validate spec dictionary before loading.
    
    Args:
        data: Dictionary representation of spec
        strict: If True, treat warnings as errors
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level keys
    if "metadata" not in data:
        errors.append("Missing required key: 'metadata'")
    elif not isinstance(data["metadata"], dict):
        errors.append("'metadata' must be a dictionary")
    else:
        # Check required metadata fields
        for field in ["id", "title", "version"]:
            if field not in data["metadata"]:
                errors.append(f"Missing required metadata field: '{field}'")
    
    # Check optional top-level keys are correct type
    type_checks = {
        "principles": list,
        "rules": list,
        "interfaces": dict,
        "glossary": list,
        "changelog": list,
    }
    
    for key, expected_type in type_checks.items():
        if key in data and not isinstance(data[key], expected_type):
            errors.append(f"'{key}' must be a {expected_type.__name__}")
    
    return errors


def validate_spec_file(path: str, strict: bool = False) -> bool:
    """Validate a spec file.
    
    Args:
        path: Path to spec JSON file
        strict: If True, treat warnings as errors
        
    Returns:
        True if validation passes
        
    Raises:
        SpecValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    from synth_ai.sdk.specs.loader import load_spec_from_file
    
    spec = load_spec_from_file(path)
    validator = SpecValidator()
    return validator.validate(spec, strict=strict)

