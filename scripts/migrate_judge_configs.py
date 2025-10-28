#!/usr/bin/env python3
"""
Migrate judge/rubric configs by removing deprecated fields.

Usage:
    python scripts/migrate_judge_configs.py path/to/config.toml
    python scripts/migrate_judge_configs.py --all  # Migrate all example configs
"""

import sys
from pathlib import Path
import shutil

try:
    import toml
except ImportError:
    print("Error: toml package not installed. Run: pip install toml")
    sys.exit(1)


def migrate_config(config: dict) -> tuple[dict, list[str]]:
    """
    Remove deprecated fields from config.
    
    Returns:
        (migrated_config, list_of_changes)
    """
    changes = []
    
    # Migrate rubric section
    if "rubric" in config:
        rubric = config["rubric"]
        
        # Remove deprecated fields
        for field in ["model", "api_base", "api_key_env"]:
            if field in rubric:
                rubric.pop(field)
                changes.append(f"Removed [rubric].{field} (deprecated)")
        
        # Remove deprecated subsections
        if "event" in rubric:
            rubric.pop("event")
            changes.append("Removed [rubric.event] section (deprecated)")
        
        if "outcome" in rubric:
            rubric.pop("outcome")
            changes.append("Removed [rubric.outcome] section (deprecated)")
    
    # Migrate judge section
    if "judge" in config:
        judge = config["judge"]
        
        # Remove deprecated top-level fields
        if "type" in judge:
            judge.pop("type")
            changes.append("Removed [judge].type (deprecated)")
        
        # Migrate timeout_s to judge.options.timeout_s
        if "timeout_s" in judge:
            timeout = judge.pop("timeout_s")
            if "options" not in judge:
                judge["options"] = {}
            if "timeout_s" not in judge["options"]:
                judge["options"]["timeout_s"] = timeout
                changes.append(f"Migrated [judge].timeout_s → [judge.options].timeout_s")
            else:
                changes.append("Removed [judge].timeout_s (already in options)")
        
        # Remove deprecated options fields
        if "options" in judge:
            options = judge["options"]
            
            for field in ["max_concurrency", "tracks"]:
                if field in options:
                    options.pop(field)
                    changes.append(f"Removed [judge.options].{field} (deprecated)")
            
            # Remove rubric_overrides if empty (usually overridden by TaskInfo)
            if "rubric_overrides" in options:
                overrides = options["rubric_overrides"]
                if not overrides or (isinstance(overrides, dict) and not any(overrides.values())):
                    options.pop("rubric_overrides")
                    changes.append("Removed empty [judge.options].rubric_overrides")
    
    return config, changes


def migrate_file(path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single TOML file.
    
    Returns:
        True if changes were made, False otherwise
    """
    try:
        # Read original
        with open(path) as f:
            config = toml.load(f)
        
        # Migrate
        migrated, changes = migrate_config(config)
        
        if not changes:
            print(f"✅ {path} - Already clean (no changes)")
            return False
        
        print(f"📝 {path} - Found {len(changes)} changes:")
        for change in changes:
            print(f"   • {change}")
        
        if dry_run:
            print(f"   (Dry run - no files modified)")
            return True
        
        # Write backup
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)
        
        # Write migrated
        with open(path, "w") as f:
            toml.dump(migrated, f)
        
        print(f"   💾 Saved (backup: {backup_path.name})")
        return True
    
    except Exception as exc:
        print(f"❌ {path} - Error: {exc}")
        return False


def find_all_configs(root: Path) -> list[Path]:
    """Find all TOML configs in examples directory."""
    configs = []
    for toml_path in root.rglob("*.toml"):
        # Skip backup files
        if toml_path.suffix == ".bak" or ".bak" in toml_path.suffixes:
            continue
        configs.append(toml_path)
    return sorted(configs)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    arg = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    
    if arg == "--all":
        # Migrate all example configs
        repo_root = Path(__file__).parent.parent
        examples_dir = repo_root / "examples"
        
        if not examples_dir.exists():
            print(f"Error: {examples_dir} not found")
            sys.exit(1)
        
        configs = find_all_configs(examples_dir)
        print(f"Found {len(configs)} config files in {examples_dir}")
        print()
        
        changed = 0
        for config_path in configs:
            if migrate_file(config_path, dry_run=dry_run):
                changed += 1
            print()
        
        print(f"{'[DRY RUN] ' if dry_run else ''}Summary: {changed}/{len(configs)} files modified")
    
    else:
        # Migrate single file
        config_path = Path(arg)
        
        if not config_path.exists():
            print(f"Error: {config_path} not found")
            sys.exit(1)
        
        migrate_file(config_path, dry_run=dry_run)


if __name__ == "__main__":
    main()

