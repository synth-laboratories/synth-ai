import shutil
from pathlib import Path

import click

DEMO_SOURCES: dict[str, str] = {
    "local": "crafter",
    "modal": "math"
}


@click.command()
@click.option(
    "--runtime",
    "runtime",
    type=click.Choice(tuple(DEMO_SOURCES.keys()), case_sensitive=False),
    default="local",
    show_default=True,
    help="Select runtime to load a demo task app to your cwd. Options: local, modal"
)
def demo_cmd(runtime: str) -> None:
      runtime_key = runtime.lower()
      demo_name = DEMO_SOURCES[runtime_key]
      package_root = Path(__file__).resolve().parents[1]
      src = package_root / "demos" / demo_name
      if not src.exists():
          raise click.ClickException(f"Demo source directory not found: {src}")

      dst = Path.cwd() / src.name
      if dst.exists():
          raise click.ClickException(
              f"Destination already exists: {dst}. Remove it first if you want to re-copy."
          )

      shutil.copytree(src, dst)
      click.echo(f"Copied {demo_name} demo to {dst}")
