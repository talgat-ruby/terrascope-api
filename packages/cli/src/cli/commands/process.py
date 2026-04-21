import asyncio
import json
from pathlib import Path

import typer
from shapely.geometry import shape as shapely_shape

from core.config import settings

app = typer.Typer(help="Process satellite imagery")


@app.callback(invoke_without_command=True)
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Path to GeoTIFF file"),
    aoi: Path | None = typer.Option(
        None, "--aoi", "-a", help="Path to AOI GeoJSON file (uses full raster extent if omitted)"
    ),
    output: Path = typer.Option("./output", "--output", "-o", help="Output directory"),
    use_temporal: bool = typer.Option(
        False, "--use-temporal", help="Submit to Temporal instead of running locally"
    ),
) -> None:
    """Process satellite imagery and extract features."""
    if use_temporal:
        asyncio.run(_run_temporal(input, aoi, output))
    else:
        _run_local(input, aoi, output)


def _run_local(input_path: Path, aoi_path: Path | None, output_dir: Path) -> None:
    """Run the full processing pipeline locally using core services."""
    from core.services.detector import DetectorService
    from core.services.exporter import GISExporterService
    from core.services.imagery import ImageryLoaderService
    from core.services.indicators import IndicatorCalculatorService
    from core.services.postprocessor import PostprocessingConfig, PostprocessorService
    from core.services.tiler import TilerService

    # Step 1: Load imagery
    typer.echo("Loading imagery...")
    loader = ImageryLoaderService()
    dataset = loader.load(str(input_path))
    try:
        metadata = loader.get_metadata(dataset)
        typer.echo(
            f"  Bands: {metadata['band_count']}, CRS: {metadata['crs']}, "
            f"Resolution: {metadata['resolution']}"
        )
        if aoi_path is not None:
            aoi_geojson = json.loads(aoi_path.read_text())
            aoi_geom = shapely_shape(aoi_geojson)
            data, transform, crs = loader.clip_to_aoi(dataset, aoi_geom)
        else:
            aoi_geom = loader.get_bounds_geometry(dataset)
            typer.echo("  No AOI provided, using full raster extent")
            data, transform, crs = loader.clip_to_aoi(dataset, aoi_geom, aoi_crs=str(dataset.crs))
    finally:
        dataset.close()
    typer.echo(f"  Clipped shape: {data.shape}")

    # Step 2: Tile imagery
    typer.echo("Tiling imagery...")
    tiler = TilerService(tile_size=settings.tile_size, overlap=settings.tile_overlap)
    tiles = list(tiler.generate_tiles(data, transform, crs=crs))
    typer.echo(f"  Generated {len(tiles)} tiles")

    # Step 3: Detect objects
    typer.echo("Running detection...")
    in_channels = data.shape[0]
    detector = DetectorService(device=settings.device)
    detector.load_models(in_channels=in_channels)
    all_detections = []
    for i, tile in enumerate(tiles):
        tile_dets = detector.predict_tile(tile)
        all_detections.extend(tile_dets)
        typer.echo(f"  Tile {i + 1}/{len(tiles)}: {len(tile_dets)} detections")
    typer.echo(f"  Total raw detections: {len(all_detections)}")

    # Step 4: Post-process
    typer.echo("Post-processing...")
    postprocessor = PostprocessorService()
    config = PostprocessingConfig(
        iou_threshold=settings.nms_iou_threshold,
        confidence_threshold=settings.confidence_threshold,
        min_area_m2=settings.min_area_m2,
        max_area_m2=settings.max_area_m2,
        simplify_tolerance_m=settings.simplify_tolerance_m,
    )
    filtered, stats = postprocessor.run(all_detections, config, aoi_geom)
    typer.echo(f"  {stats['input_count']} -> {stats['output_count']} detections")

    # Step 5: Export
    typer.echo("Exporting results...")
    exporter = GISExporterService()
    paths = exporter.export_all(filtered, crs, output_dir)
    for fmt, path in paths.items():
        typer.echo(f"  {fmt}: {path}")

    # Step 6: Compute indicators (use AOI as a single zone)
    typer.echo("Computing indicators...")
    calculator = IndicatorCalculatorService()
    zones = {"aoi": aoi_geom}
    indicators = calculator.compute(filtered, zones)
    for ind in indicators:
        typer.echo(
            f"  {ind.zone_id}/{ind.class_name}: "
            f"count={ind.count}, density={ind.density_per_km2}/km2, "
            f"area={ind.total_area_m2:.1f} m2"
        )

    # Export indicators
    indicators_dir = output_dir / "indicators"
    indicators_dir.mkdir(parents=True, exist_ok=True)
    calculator.export_csv(indicators, indicators_dir / "indicators.csv")
    calculator.export_json(indicators, indicators_dir / "indicators.json")
    typer.echo(f"  Indicators exported to {indicators_dir}")

    typer.echo("Done!")


async def _run_temporal(input_path: Path, aoi_path: Path | None, output_dir: Path) -> None:
    """Submit processing to Temporal workflow."""
    from temporalio.client import Client

    from core.database import async_session_factory
    from core.models.processing import ProcessingJob
    from worker.workflows.processing import ProcessingWorkflow

    aoi_geojson = json.loads(aoi_path.read_text()) if aoi_path is not None else None

    async with async_session_factory() as session:
        job = ProcessingJob(
            input_path=str(input_path),
            config={
                "aoi": aoi_geojson,
                "aoi_crs": "EPSG:4326",
            },
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
    typer.echo("Waiting for completion...")

    result = await handle.result()
    typer.echo(f"Workflow completed: {result['status']}")
    typer.echo(f"Output directory: {output_dir}")
