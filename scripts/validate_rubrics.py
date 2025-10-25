#!/usr/bin/env python3
"""Validate all rubric files in the repository.

This script checks:
1. Task app rubrics (*.json, not *_backend_judge.json) match the Rubric schema
2. Backend judge rubrics (*_backend_judge.json) have event/outcome structure
3. All rubrics have correct weight distributions

Run this before committing changes to rubric files.
"""

import json
import sys
from pathlib import Path

def validate_task_app_rubric(path: Path) -> tuple[bool, list[str]]:
    """Validate a task app rubric file."""
    errors = []
    
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    
    # Check required fields
    required = ["version", "goal_text", "aggregation", "criteria"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if "criteria" in data:
        criteria = data["criteria"]
        if not isinstance(criteria, list):
            errors.append("'criteria' must be a list")
        elif len(criteria) == 0:
            errors.append("'criteria' cannot be empty")
        else:
            # Check weight sum
            total_weight = sum(c.get("weight", 0) for c in criteria)
            if abs(total_weight - 1.0) > 1e-6:
                errors.append(f"Criteria weights sum to {total_weight}, expected 1.0")
            
            # Validate each criterion
            for i, criterion in enumerate(criteria):
                if "id" not in criterion:
                    errors.append(f"Criterion {i} missing 'id'")
                if "description" not in criterion:
                    errors.append(f"Criterion {i} missing 'description'")
                if "weight" not in criterion:
                    errors.append(f"Criterion {i} missing 'weight'")
    
    return len(errors) == 0, errors


def validate_backend_judge_rubric(path: Path) -> tuple[bool, list[str]]:
    """Validate a backend judge rubric file."""
    errors = []
    
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    
    # Check required fields
    if "event" not in data:
        errors.append("Missing required field: 'event'")
    if "outcome" not in data:
        errors.append("Missing required field: 'outcome'")
    
    # Validate structure
    for section in ["event", "outcome"]:
        if section in data:
            if not isinstance(data[section], list):
                errors.append(f"'{section}' must be a list")
            else:
                for i, criterion in enumerate(data[section]):
                    required_fields = ["id", "description", "weight", "scale"]
                    for field in required_fields:
                        if field not in criterion:
                            errors.append(f"{section}[{i}] missing '{field}'")
                    
                    if "scale" in criterion:
                        if criterion["scale"] not in ["bounded", "unbounded"]:
                            errors.append(
                                f"{section}[{i}] 'scale' must be 'bounded' or 'unbounded', "
                                f"got {criterion['scale']!r}"
                            )
    
    return len(errors) == 0, errors


def main():
    """Run validation on all rubric files."""
    rubrics_dir = Path("examples/multi_step/rubrics")
    
    if not rubrics_dir.exists():
        print(f"❌ Rubrics directory not found: {rubrics_dir}")
        return 1
    
    rubric_files = sorted(rubrics_dir.glob("*.json"))
    
    if not rubric_files:
        print(f"⚠️  No rubric files found in {rubrics_dir}")
        return 0
    
    print(f"Validating {len(rubric_files)} rubric files...\n")
    
    failed = []
    passed = []
    
    for rubric_file in rubric_files:
        is_backend_judge = rubric_file.name.endswith("_backend_judge.json")
        
        if is_backend_judge:
            success, errors = validate_backend_judge_rubric(rubric_file)
            rubric_type = "backend judge"
        else:
            success, errors = validate_task_app_rubric(rubric_file)
            rubric_type = "task app"
        
        if success:
            print(f"✅ {rubric_file.name} ({rubric_type})")
            passed.append(rubric_file.name)
        else:
            print(f"❌ {rubric_file.name} ({rubric_type})")
            for error in errors:
                print(f"   - {error}")
            failed.append(rubric_file.name)
    
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    print(f"{'='*60}")
    
    if failed:
        print("\n❌ Some rubrics failed validation:")
        for name in failed:
            print(f"   - {name}")
        return 1
    else:
        print("\n✅ All rubrics are valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

