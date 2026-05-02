"""Local entry point for the detection pipeline."""

import json
from pathlib import Path

import typer
from shapely.geometry import shape as shapely_shape

from core.config import settings

app = typer.Typer(help="Detect objects on satellite imagery")


@app.callback(invoke_without_command=True)
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Path to GeoTIFF"),
    aoi: Path | None = typer.Option(
        None, "--aoi", "-a", help="GeoJSON AOI (defaults to full raster)"
    ),
    output: Path = typer.Option("./output", "--output", "-o"),
    detectors: str = typer.Option(
        ...,
        "--detectors",
        help=(
            "JSON list of detector specs, e.g. "
            '\'[{"name":"yolov8-obb-aerial","classes":["ship"]},'
            '{"name":"segformer-landscape","classes":["building","road"],'
            '"min_confidence":0.5}]\''
        ),
    ),
) -> None:
    parsed = json.loads(detectors)
    _run_local(input, aoi, output, parsed)


def _run_local(
    input_path: Path,
    aoi_path: Path | None,
    output_dir: Path,
    detector_specs: list[dict],
) -> None:
    from core.detection import (
        DetectorSpec,
        build_from_specs,
        filter_detections,
        render_overlay,
    )
    from core.services.exporter import GISExporterService
    from core.services.imagery import ImageryLoaderService
    from core.services.indicators import IndicatorCalculatorService

    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_geom = (
        shapely_shape(json.loads(aoi_path.read_text()))
        if aoi_path is not None
        else None
    )

    typer.echo(f"Loading imagery from {input_path}...")
    raster = ImageryLoaderService().load_clipped(input_path, aoi_geom)
    typer.echo(f"  Shape: {raster.data.shape}, CRS: {raster.crs}")

    typer.echo("Running detection...")
    specs = DetectorSpec.list_from_config({"detectors": detector_specs})
    det = build_from_specs(specs)
    detections = det.detect(raster)
    names = ", ".join(s.name for s in specs)
    typer.echo(f"  Raw detections: {len(detections)} via [{names}]")

    detections = filter_detections(
        detections,
        min_confidence=settings.min_confidence,
        aoi=raster.aoi_geom,
        specs=specs,
    )
    typer.echo(f"  After filter: {len(detections)}")

    overlay_path = output_dir / "overlay.png"
    render_overlay(raster, detections, overlay_path)
    typer.echo(f"  PNG overlay: {overlay_path}")

    geojson_path = output_dir / "detections.geojson"
    GISExporterService().export_geojson(detections, geojson_path, crs=raster.crs)
    typer.echo(f"  GeoJSON: {geojson_path}")

    if raster.aoi_geom is not None:
        calculator = IndicatorCalculatorService()
        indicators = calculator.compute(detections, {"aoi": raster.aoi_geom})
        if indicators:
            indicators_dir = output_dir / "indicators"
            calculator.export_csv(indicators, indicators_dir / "indicators.csv")
            calculator.export_json(indicators, indicators_dir / "indicators.json")
            typer.echo(f"  Indicators: {indicators_dir}")

    typer.echo("Done.")
