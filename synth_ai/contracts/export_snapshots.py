"""Update canonical source/wheel schemas with the pinned protocol renderer."""

import argparse
from pathlib import Path

import pydantic

from .schema import export_contract as index_contract
from .swarms_schema import export_contract as swarms_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if pydantic.__version__ != "2.12.5":
        raise SystemExit("Canonical export requires the pinned Pydantic 2.12.5 renderer")
    root = Path(__file__).parents[2]
    if not (root / "pyproject.toml").is_file():
        raise SystemExit("Canonical export requires the SDK source checkout")
    for name, export in (("index-v1.json", index_contract), ("swarms-v1.json", swarms_contract)):
        content = export()
        for folder in (root / "contracts", Path(__file__).parent / "snapshots"):
            target = folder / name
            if args.check:
                if not target.is_file() or target.read_text() != content:
                    raise SystemExit("Canonical snapshot update required: " + name)
            else:
                folder.mkdir(exist_ok=True)
                target.write_text(content)


if __name__ == "__main__":
    main()
