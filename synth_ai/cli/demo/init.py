from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import click
from synth_ai.cli.lib import ensure_modal_installed
from synth_ai.cli.lib.print_next_step_message import print_next_step_message
from synth_ai.demo_registry import DemoTemplate, get_demo_template, list_demo_templates
from synth_ai.demos import core as demo_core


def run_init(template: str | None, dest: str | None, force: bool) -> int:
    """Materialise a demo task app template into the current directory."""

    ## Load available templates
    templates = list(list_demo_templates())
    if not templates:
        print("No demo templates registered. Update synth_ai/demo_registry.py to add entries.")
        return 1

    ## Resolve selected template (CLI arg or prompt)
    selected: DemoTemplate | None = None
    if template:
        selected = get_demo_template(template)
        if selected is None:
            available = ", ".join(t.template_id for t in templates)
            print(f"Unknown template '{template}'. Available: {available}")
            return 1
    else:
        print("\nWelcome to the Synth AI SDK demo!")
        print("\nStart by selecting a task app:")
        for idx, entry in enumerate(templates, start=1):
            print(f"  [{idx}] {entry.name} ({entry.template_id})")
            print(f"      {entry.description}")
        try:
            choice_raw = input(f"Enter choice [1-{len(templates)}] (default 1): ").strip() or "1"
        except Exception:
            choice_raw = "1"
        if not choice_raw.isdigit():
            print("Selection must be a number.")
            return 1
        choice_idx = int(choice_raw)
        if not 1 <= choice_idx <= len(templates):
            print("Selection out of range.")
            return 1
        selected = templates[choice_idx - 1]

    assert selected is not None

    ## Determine destination directory
    default_subdir = selected.default_subdir or selected.template_id
    if dest:
        default_dest = Path(dest).expanduser().resolve()
    else:
        default_dest = (Path.cwd() / default_subdir).resolve()
    if dest:
        destination = Path(dest).expanduser().resolve()
    else:
        try:
            dest_input = input(f"\nWhere do you want to store this task app?\nPress enter for default path: {default_dest}\n> ").strip()
        except Exception:
            dest_input = ""
        destination = Path(dest_input).expanduser().resolve() if dest_input else default_dest
    directory_cleared = False

    ## Prepare destination directory (clear or create)
    if destination.exists():
        if destination.is_file():
            print(f"Destination {destination} is a file. Provide a directory path.")
            return 1
        if any(destination.iterdir()) and not force:
            try:
                response = (
                    input(f"{destination} is not empty\nOverwrite? [y/N]: ")
                    .strip()
                    .lower()
                )
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 1
            if response not in ("y", "yes"):
                print("Cancelled. Choose another directory or delete the existing one.")
                return 1
        print(f"Clearing {destination}...")
        try:
            shutil.rmtree(destination)
        except Exception as exc:
            print(f"Error clearing directory: {exc}")
            print("Please manually remove the directory and try again.")
            return 1
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            print(f"Warning: Directory {destination} still contains files after clearing.")
            print("Some files may not have been removed. Please check manually.")
            return 1
        directory_cleared = True
    else:
        destination.mkdir(parents=True, exist_ok=True)

    ## Ensure Modal CLI installed when template requires it
    if selected.requires_modal:
        ensure_modal_installed()

    try:
        ## Copy template files into destination
        for spec in selected.iter_copy_specs():
            src_path = spec.absolute_source()
            if not src_path.exists():
                print(f"Template source missing: {src_path}")
                return 1
            dest_path = (destination / spec.destination).resolve()

            if src_path.is_dir():
                if dest_path.exists() and not directory_cleared:
                    try:
                        response = (
                            input(f"Directory {dest_path.name} exists. Overwrite? [y/N]: ")
                            .strip()
                            .lower()
                        )
                    except (EOFError, KeyboardInterrupt):
                        print("\nCancelled.")
                        return 1
                    if response not in ("y", "yes"):
                        print(f"Skipping {dest_path.name}")
                        continue
                    shutil.rmtree(dest_path)
                elif dest_path.exists() and directory_cleared:
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists() and not directory_cleared and not force:
                    try:
                        response = (
                            input(f"File {dest_path.name} exists. Overwrite? [y/N]: ")
                            .strip()
                            .lower()
                        )
                    except (EOFError, KeyboardInterrupt):
                        print("\nCancelled.")
                        return 1
                    if response not in ("y", "yes"):
                        print(f"Skipping {dest_path.name}")
                        continue
                shutil.copy2(src_path, dest_path)
                if spec.make_executable:
                    try:
                        st_mode = os.stat(dest_path).st_mode
                        os.chmod(dest_path, st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    except Exception:
                        pass

        ## Store template environment defaults if provided
        if selected.env_lines:
            defaults: dict[str, str] = {}
            for line in selected.env_lines:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                defaults[key.strip()] = value.strip()
            if defaults:
                demo_core.persist_dotenv_values(defaults, cwd=str(destination))

        ## Copy default config if present
        config_src = selected.config_source_path()
        if config_src and config_src.exists():
            cfg_dst = (destination / selected.config_destination).resolve()
            should_copy = directory_cleared or force or not cfg_dst.exists()
            if cfg_dst.exists() and not (directory_cleared or force):
                try:
                    response = (
                        input(f"File {cfg_dst.name} exists. Overwrite? [y/N]: ").strip().lower()
                    )
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return 1
                should_copy = response in ("y", "yes")
            if should_copy:
                cfg_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_src, cfg_dst)
            else:
                print(f"Skipping {cfg_dst.name}")

        ## Execute any template post-processing hook
        if selected.post_copy is not None:
            try:
                selected.post_copy(destination)
            except Exception as post_exc:
                print(f"Post-processing failed: {post_exc}")
                return 1

        demo_core.record_task_app(str(destination), template_id=selected.template_id)
        demo_core.persist_demo_dir(str(destination))
        demo_core.persist_template_id(selected.template_id)

        ## Summarise results
        print(f"\nDemo files created at\n{destination}")
        for spec in selected.iter_copy_specs():
            print(f" - {spec.destination}")
        if selected.env_lines:
            print(" - environment defaults stored")
        if selected.config_source_path():
            print(f" - {selected.config_destination}")

        # Mark this shell session as running the demo flows
        os.environ["DEMO_MODE"] = "1"
        os.environ["DEMO_DIR"] = str(destination)
        print("\nEnvironment flags set for this session:")
        print("  DEMO_MODE=1")
        print(f"  DEMO_DIR={destination}")

        print_next_step_message("set up your environment", ["uvx synth-ai setup"])
        return 0
    except KeyboardInterrupt:
        return 1
    except Exception as exc:
        print(f"Init failed: {exc}")
        return 1


def register(group):
    @group.command("init")
    @click.option("--template", type=str, default=None, help="Template id to instantiate")
    @click.option("--dest", type=str, default=None, help="Destination directory for files")
    @click.option("--force", is_flag=True, help="Overwrite existing files in destination")
    def demo_init(template: str | None, dest: str | None, force: bool):
        code = run_init(template=template, dest=dest, force=force)
        if code:
            raise click.exceptions.Exit(code)
