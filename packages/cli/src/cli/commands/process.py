"""Local + Temporal entry points for the detection pipeline."""

import asyncio
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
    detector: str = typer.Option(
        None,
        "--detector",
        help="Detector name (default from settings.detector_name)",
    ),
    use_temporal: bool = typer.Option(
        False,
        "--use-temporal",
        help="Submit job to Temporal instead of running locally",
    ),
) -> None:
    if use_temporal:
        asyncio.run(_run_temporal(input, aoi, output))
    else:
        _run_local(input, aoi, output, detector)


def _run_local(
    input_path: Path, aoi_path: Path | None, output_dir: Path, detector_name: str | None
) -> None:
    from core.detection import build_detector, filter_detections, render_overlay
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
    det = build_detector(detector_name)
    detections = det.detect(raster)
    typer.echo(f"  Raw detections: {len(detections)} via {det.name}")

    detections = filter_detections(
        detections, min_confidence=settings.min_confidence, aoi=raster.aoi_geom
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


async def _run_temporal(
    input_path: Path, aoi_path: Path | None, output_dir: Path
) -> None:
    from temporalio.client import Client

    from core.database import async_session_factory
    from core.models.processing import ProcessingJob
    from worker.workflows.processing import ProcessingWorkflow

    aoi_geojson = json.loads(aoi_path.read_text()) if aoi_path is not None else None

    async with async_session_factory() as session:
        job = ProcessingJob(
            input_path=str(input_path),
            config={"aoi": aoi_geojson, "aoi_crs": "EPSG:4326"},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = str(job.id)

    typer.echo(f"Created job: {job_id}")

    client = await Client.connect(settings.temporal_address)
    handle = await client.start_workflow(
        ProcessingWorkflow.run,
        job_id,
        id=f"processing-{job_id}",
        task_queue=settings.temporal_task_queue,
    )
    typer.echo(f"Started workflow: {handle.id}")
    result = await handle.result()
    typer.echo(f"Workflow completed: {result['status']}")
    typer.echo(f"Output directory: {output_dir}")
