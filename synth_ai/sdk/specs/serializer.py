"""Serializers for converting specs to prompt-friendly formats."""

from __future__ import annotations

from typing import Optional

from synth_ai.sdk.specs.dataclasses import Spec


def spec_to_prompt_context(
    spec: Spec,
    include_examples: bool = True,
    include_tests: bool = False,
    include_glossary: bool = True,
    max_rules: Optional[int] = None,
    priority_threshold: Optional[int] = None,
) -> str:
    """Convert a Spec to a prompt-friendly markdown format.
    
    Args:
        spec: The specification to serialize
        include_examples: Whether to include rule examples
        include_tests: Whether to include test cases
        include_glossary: Whether to include glossary terms
        max_rules: Maximum number of rules to include (None = all)
        priority_threshold: Only include rules with priority >= threshold
        
    Returns:
        Markdown-formatted string suitable for inclusion in prompts
    """
    lines = []
    
    # Header
    lines.append(f"# {spec.metadata.title}")
    if spec.metadata.description:
        lines.append(f"\n{spec.metadata.description}")
    lines.append(f"\n**Version:** {spec.metadata.version}")
    if spec.metadata.scope:
        lines.append(f"**Scope:** {spec.metadata.scope}")
    lines.append("")
    
    # Principles
    if spec.principles:
        lines.append("## Guiding Principles\n")
        for principle in spec.principles:
            lines.append(f"**{principle.id}**: {principle.text}")
            if principle.rationale:
                lines.append(f"  - *Rationale:* {principle.rationale}")
            lines.append("")
    
    # Rules
    if spec.rules:
        # Filter rules by priority if specified
        rules_to_include = spec.rules
        if priority_threshold is not None:
            rules_to_include = [
                r for r in rules_to_include
                if r.priority is not None and r.priority >= priority_threshold
            ]
        
        # Sort by priority (highest first)
        rules_to_include = sorted(
            rules_to_include,
            key=lambda r: r.priority if r.priority is not None else 0,
            reverse=True,
        )
        
        # Limit number of rules
        if max_rules is not None:
            rules_to_include = rules_to_include[:max_rules]
        
        if rules_to_include:
            lines.append("## Rules and Policies\n")
            
            for rule in rules_to_include:
                # Rule header
                priority_str = f" [Priority: {rule.priority}]" if rule.priority else ""
                lines.append(f"### {rule.id}: {rule.title}{priority_str}\n")
                
                if rule.rationale:
                    lines.append(f"*Rationale:* {rule.rationale}\n")
                
                # Constraints
                if rule.constraints.must or rule.constraints.must_not:
                    lines.append("**Constraints:**")
                    
                    if rule.constraints.must:
                        lines.append("- **MUST:**")
                        for constraint in rule.constraints.must:
                            lines.append(f"  - {constraint}")
                    
                    if rule.constraints.must_not:
                        lines.append("- **MUST NOT:**")
                        for constraint in rule.constraints.must_not:
                            lines.append(f"  - {constraint}")
                    
                    if rule.constraints.should:
                        lines.append("- **SHOULD:**")
                        for constraint in rule.constraints.should:
                            lines.append(f"  - {constraint}")
                    
                    if rule.constraints.should_not:
                        lines.append("- **SHOULD NOT:**")
                        for constraint in rule.constraints.should_not:
                            lines.append(f"  - {constraint}")
                    
                    lines.append("")
                
                # Examples
                if include_examples and rule.examples:
                    lines.append("**Examples:**\n")
                    
                    good_examples = [e for e in rule.examples if e.kind == "good"]
                    bad_examples = [e for e in rule.examples if e.kind == "bad"]
                    
                    if good_examples:
                        lines.append("✅ **Good:**")
                        for ex in good_examples:
                            lines.append(f"- Prompt: \"{ex.prompt}\"")
                            lines.append(f"  Response: \"{ex.response}\"")
                            if ex.description:
                                lines.append(f"  *{ex.description}*")
                        lines.append("")
                    
                    if bad_examples:
                        lines.append("❌ **Bad:**")
                        for ex in bad_examples:
                            lines.append(f"- Prompt: \"{ex.prompt}\"")
                            lines.append(f"  Response: \"{ex.response}\"")
                            if ex.description:
                                lines.append(f"  *{ex.description}*")
                        lines.append("")
                
                # Tests
                if include_tests and rule.tests:
                    lines.append("**Test Cases:**\n")
                    for test in rule.tests:
                        lines.append(f"- {test.id}: {test.challenge}")
                        if test.asserts:
                            lines.append(f"  Asserts: {', '.join(test.asserts)}")
                        if test.expected_behavior:
                            lines.append(f"  Expected: {test.expected_behavior}")
                    lines.append("")
    
    # Glossary
    if include_glossary and spec.glossary:
        lines.append("## Glossary\n")
        for item in spec.glossary:
            aliases_str = f" (aliases: {', '.join(item.aliases)})" if item.aliases else ""
            lines.append(f"**{item.term}**{aliases_str}: {item.definition}")
        lines.append("")
    
    return "\n".join(lines)


def spec_to_compact_context(spec: Spec, max_tokens: int = 5000) -> str:
    """Convert a Spec to a compact prompt context within token limit.
    
    Prioritizes high-priority rules and essential information.
    
    Args:
        spec: The specification to serialize
        max_tokens: Approximate maximum tokens (uses char estimation: 4 chars ≈ 1 token)
        
    Returns:
        Compact markdown-formatted string
    """
    # Start with high-priority rules only
    context = spec_to_prompt_context(
        spec,
        include_examples=True,
        include_tests=False,
        include_glossary=True,
        priority_threshold=7,
    )
    
    # If still too long, try with fewer examples
    max_chars = max_tokens * 4
    if len(context) > max_chars:
        context = spec_to_prompt_context(
            spec,
            include_examples=False,
            include_tests=False,
            include_glossary=True,
            priority_threshold=7,
        )
    
    # If still too long, reduce glossary
    if len(context) > max_chars:
        context = spec_to_prompt_context(
            spec,
            include_examples=False,
            include_tests=False,
            include_glossary=False,
            priority_threshold=8,
        )
    
    return context

