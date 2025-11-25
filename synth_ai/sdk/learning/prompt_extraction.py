"""Unified abstraction for extracting prompt content from GEPA/MIPRO results.

This module provides a clean, type-safe API for extracting prompt text from
various serialized structures returned by prompt learning jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedPrompt:
    """Extracted prompt content with metadata."""
    
    text: str
    role: str = "system"
    sections: List[Dict[str, Any]] = None  # type: ignore[assignment]
    text_replacements: List[Dict[str, Any]] = None  # type: ignore[assignment]
    example_injections: List[Dict[str, Any]] = None  # type: ignore[assignment]
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.sections is None:
            self.sections = []
        if self.text_replacements is None:
            self.text_replacements = []
        if self.example_injections is None:
            self.example_injections = []
    
    def to_formatted_string(self) -> str:
        """Format prompt as a readable string."""
        parts = []
        
        # Add main text with role
        if self.text:
            parts.append(f"[{self.role.upper()}]\n{self.text}")
        
        # Add example injections if present
        if self.example_injections:
            parts.append(f"\n[EXAMPLES] ({len(self.example_injections)} few-shot examples)")
            for i, example_inj in enumerate(self.example_injections[:3], 1):
                if isinstance(example_inj, dict):
                    examples = example_inj.get("examples", [])
                    if examples:
                        parts.append(f"\n  Example {i}:")
                        for ex in examples[:2]:
                            if isinstance(ex, dict):
                                role = ex.get("role", "user")
                                content = ex.get("content", "")
                                if content:
                                    preview = content[:100] + "..." if len(content) > 100 else content
                                    parts.append(f"    {role}: {preview}")
        
        return "\n".join(parts)


class PromptExtractor:
    """Extracts prompt content from typed candidate dataclasses or raw dictionaries."""
    
    @staticmethod
    def extract_from_candidate(
        candidate: Any,  # OptimizedCandidate, AttemptedCandidate, or Dict[str, Any]
    ) -> Optional[ExtractedPrompt]:
        """Extract prompt content from a candidate (typed or raw dict).
        
        Args:
            candidate: OptimizedCandidate/AttemptedCandidate instance or dict
            
        Returns:
            ExtractedPrompt instance or None if extraction fails
        """
        from .prompt_learning_types import OptimizedCandidate
        
        # Handle typed dataclasses
        if isinstance(candidate, OptimizedCandidate):
            transformation = candidate.get_transformation()
            if transformation:
                return PromptExtractor._extract_from_transformation_obj(transformation)
            
            template = candidate.get_template()
            if template:
                return PromptExtractor._extract_from_template_obj(template)
            
            # Fallback to raw object dict
            obj = candidate.object
            payload_kind = candidate.payload_kind
        elif hasattr(candidate, "object") and hasattr(candidate, "payload_kind"):  # AttemptedCandidate-like
            obj = candidate.object or {}
            payload_kind = candidate.payload_kind
        elif isinstance(candidate, dict):
            # Handle raw dict (backward compatibility)
            obj = candidate.get("object", {})
            payload_kind = candidate.get("payload_kind", "")
        else:
            return None
        
        if not isinstance(obj, dict):
            return None
        
        # Try transformation format first (most common for GEPA)
        if payload_kind == "transformation" or "text_replacements" in obj:
            return PromptExtractor._extract_from_transformation(obj)
        
        # Try template format
        if payload_kind == "template" or "sections" in obj:
            return PromptExtractor._extract_from_template(obj)
        
        # Fallback: try nested structures
        data = obj.get("data", {})
        if isinstance(data, dict):
            if "text_replacements" in data:
                return PromptExtractor._extract_from_transformation(data)
            if "sections" in data:
                return PromptExtractor._extract_from_template({"sections": data["sections"]})
        
        return None
    
    @staticmethod
    def _extract_from_transformation_obj(transformation: Any) -> Optional[ExtractedPrompt]:
        """Extract from PromptTransformation dataclass."""
        # PromptTransformation doesn't exist in prompt_learning_types, check for attributes instead
        if not hasattr(transformation, "text_replacements"):
            return None
        
        if not transformation.text_replacements:
            return None
        
        # Use first replacement as main text
        first_replacement = transformation.text_replacements[0]
        main_text = first_replacement.new_text
        main_role = first_replacement.apply_to_role
        
        # Convert to dict format for compatibility
        text_replacements_dict = [
            {
                "new_text": tr.new_text,
                "apply_to_role": tr.apply_to_role,
                "old_text": tr.old_text,
            }
            for tr in transformation.text_replacements
        ]
        
        example_injections_dict = [
            {
                "examples": ei.examples,
                "insert_after_role": ei.insert_after_role,
            }
            for ei in transformation.example_injections
        ]
        
        return ExtractedPrompt(
            text=main_text,
            role=main_role,
            text_replacements=text_replacements_dict,
            example_injections=example_injections_dict,
        )
    
    @staticmethod
    def _extract_from_template_obj(template: Any) -> Optional[ExtractedPrompt]:
        """Extract from PromptTemplate dataclass."""
        # PromptTemplate doesn't exist in prompt_learning_types, check for attributes instead
        if not hasattr(template, "sections") or not template.sections:
            return None
        
        # Combine all section content
        text_parts = []
        role = "system"
        
        for section in template.sections:
            if section.content:
                text_parts.append(section.content)
                if not text_parts:  # Use first section's role
                    role = section.role
        
        if not text_parts:
            return None
        
        combined_text = "\n\n".join(text_parts)
        
        # Convert to dict format for compatibility
        sections_dict = [
            {
                "role": sec.role,
                "content": sec.content,
                "name": sec.name,
            }
            for sec in template.sections
        ]
        
        return ExtractedPrompt(
            text=combined_text,
            role=role,
            sections=sections_dict,
        )
    
    @staticmethod
    def _extract_from_transformation(obj: Dict[str, Any]) -> Optional[ExtractedPrompt]:
        """Extract from PromptTransformation structure.
        
        Handles both top-level and nested under 'data' structures.
        """
        # Get text_replacements (try top-level first, then nested)
        text_replacements = obj.get("text_replacements", [])
        if not text_replacements:
            data = obj.get("data", {})
            if isinstance(data, dict):
                text_replacements = data.get("text_replacements", [])
        
        if not text_replacements or not isinstance(text_replacements, list):
            return None
        
        # Extract text from first replacement (most common case)
        main_text = ""
        main_role = "system"
        all_replacements = []
        
        for replacement in text_replacements:
            if not isinstance(replacement, dict):
                continue
            
            new_text = replacement.get("new_text", "")
            role = replacement.get("apply_to_role", "system")
            
            if new_text:
                all_replacements.append(replacement)
                # Use first non-empty replacement as main text
                if not main_text:
                    main_text = new_text
                    main_role = role
        
        if not main_text:
            return None
        
        # Get example injections
        example_injections = obj.get("example_injections", [])
        if not example_injections:
            data = obj.get("data", {})
            if isinstance(data, dict):
                example_injections = data.get("example_injections", [])
        
        return ExtractedPrompt(
            text=main_text,
            role=main_role,
            text_replacements=all_replacements,
            example_injections=example_injections if isinstance(example_injections, list) else [],
        )
    
    @staticmethod
    def _extract_from_template(obj: Dict[str, Any]) -> Optional[ExtractedPrompt]:
        """Extract from PromptTemplate structure."""
        sections = obj.get("sections", [])
        if not sections:
            data = obj.get("data", {})
            if isinstance(data, dict):
                sections = data.get("sections", [])
        
        if not sections or not isinstance(sections, list):
            return None
        
        # Combine all section content
        text_parts = []
        role = "system"
        
        for section in sections:
            if not isinstance(section, dict):
                continue
            
            sec_content = section.get("content", "")
            sec_role = section.get("role", "system")
            
            if sec_content:
                text_parts.append(sec_content)
                # Use first section's role
                if not text_parts:
                    role = sec_role
        
        if not text_parts:
            return None
        
        combined_text = "\n\n".join(text_parts)
        
        return ExtractedPrompt(
            text=combined_text,
            role=role,
            sections=sections,
        )
    
    @staticmethod
    def extract_all_from_candidates(
        candidates: List[Any],  # List[OptimizedCandidate], List[AttemptedCandidate], or List[Dict]
    ) -> List[tuple[int, ExtractedPrompt]]:
        """Extract prompts from a list of candidates (typed or raw).
        
        Args:
            candidates: List of candidate dataclasses or dictionaries
            
        Returns:
            List of (index, ExtractedPrompt) tuples for successfully extracted prompts
        """
        results = []
        for idx, candidate in enumerate(candidates):
            extracted = PromptExtractor.extract_from_candidate(candidate)
            if extracted:
                results.append((idx, extracted))
        return results


def extract_prompt_text(candidate: Dict[str, Any]) -> Optional[str]:
    """Convenience function to extract prompt text from a candidate.
    
    Args:
        candidate: Candidate dictionary
        
    Returns:
        Prompt text string or None if extraction fails
    """
    extracted = PromptExtractor.extract_from_candidate(candidate)
    return extracted.text if extracted else None


def extract_prompt_formatted(candidate: Dict[str, Any]) -> Optional[str]:
    """Convenience function to extract formatted prompt from a candidate.
    
    Args:
        candidate: Candidate dictionary
        
    Returns:
        Formatted prompt string or None if extraction fails
    """
    extracted = PromptExtractor.extract_from_candidate(candidate)
    return extracted.to_formatted_string() if extracted else None

