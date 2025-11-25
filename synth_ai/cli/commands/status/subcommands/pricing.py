from __future__ import annotations

import click
from rich.table import Table

from synth_ai.core.pricing.model_pricing import MODEL_PRICES  # type: ignore[import-untyped]

from ..formatters import console


@click.command("pricing", help="List supported provider/model rates (SDK static table).")
def pricing_command() -> None:
    table = Table(title="Supported Models and Rates (USD/token)")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("Input USD", justify="right")
    table.add_column("Output USD", justify="right")
    for provider, models in MODEL_PRICES.items():
        for model, rates in models.items():
            table.add_row(provider, model, f"{rates.input_usd:.9f}", f"{rates.output_usd:.9f}")
    console.print(table)


