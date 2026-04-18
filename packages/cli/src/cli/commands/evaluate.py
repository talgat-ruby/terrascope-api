from pathlib import Path

import typer

app = typer.Typer(help="Evaluate detection quality")


@app.callback(invoke_without_command=True)
def run(
    predictions: Path = typer.Option(
        ..., "--predictions", "-p", help="Path to predictions GeoJSON"
    ),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", "-g", help="Path to ground truth GeoJSON"
    ),
    iou_threshold: float = typer.Option(
        0.5, "--iou", help="IoU threshold for matching"
    ),
) -> None:
    """Evaluate predictions against ground truth."""
    # TODO: Implement in Phase 9
    typer.echo(f"Evaluating {predictions} vs {ground_truth} (IoU={iou_threshold})")
    typer.echo("Not yet implemented -- coming in Phase 9")
