from pathlib import Path

import geopandas as gpd
import typer

from core.services.detector import RawDetection
from core.services.evaluator import QualityEvaluatorService

app = typer.Typer(help="Evaluate detection quality")


def _load_detections(path: Path) -> list[RawDetection]:
    """Load detections from a GeoJSON file into RawDetection list."""
    gdf = gpd.read_file(path)
    detections: list[RawDetection] = []
    for _, row in gdf.iterrows():
        detections.append(
            RawDetection(
                class_name=row.get("class_name", row.get("class", "unknown")),
                confidence=int(row.get("confidence", 100)),
                geometry=row.geometry,
                source=row.get("source", "ground_truth"),
            )
        )
    return detections


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
    output: Path = typer.Option(
        None, "--output", "-o", help="Path to write JSON report"
    ),
    sample_size: int = typer.Option(
        0, "--sample", help="Create control sample of N detections"
    ),
) -> None:
    """Evaluate predictions against ground truth."""
    typer.echo(f"Loading predictions from {predictions}...")
    pred_dets = _load_detections(predictions)
    typer.echo(f"  {len(pred_dets)} predictions")

    typer.echo(f"Loading ground truth from {ground_truth}...")
    gt_dets = _load_detections(ground_truth)
    typer.echo(f"  {len(gt_dets)} ground truth features")

    evaluator = QualityEvaluatorService()
    result = evaluator.evaluate(pred_dets, gt_dets, iou_threshold=iou_threshold)

    typer.echo(f"\nResults (IoU threshold={iou_threshold}):")
    typer.echo(
        f"{'Class':<15} {'Prec':>6} {'Rec':>6} {'F1':>6} {'IoU':>6} {'mAP':>6} {'TP':>4} {'FP':>4} {'FN':>4}"
    )
    typer.echo("-" * 75)
    for m in result.metrics:
        typer.echo(
            f"{m.class_name:<15} {m.precision:>6.3f} {m.recall:>6.3f} "
            f"{m.f1:>6.3f} {m.iou:>6.3f} {m.map:>6.3f} "
            f"{m.true_positives:>4} {m.false_positives:>4} {m.false_negatives:>4}"
        )

    if result.error_examples:
        typer.echo(f"\n{len(result.error_examples)} error examples found")

    if output:
        report_path = evaluator.generate_report(result, output)
        typer.echo(f"\nReport saved to {report_path}")

    if sample_size > 0:
        sample = evaluator.create_control_sample(pred_dets, sample_size, seed=42)
        typer.echo(f"\nControl sample: {len(sample)} detections selected for review")
