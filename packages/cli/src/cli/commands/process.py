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
        None,
        "--aoi",
        "-a",
        help="Path to AOI GeoJSON file (uses full raster extent if omitted)",
    ),
    output: Path = typer.Option("./output", "--output", "-o", help="Output directory"),
    use_temporal: bool = typer.Option(
        False, "--use-temporal", help="Submit to Temporal instead of running locally"
    ),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Ignore existing checkpoints and start fresh"
    ),
) -> None:
    """Process satellite imagery and extract features."""
    if use_temporal:
        asyncio.run(_run_temporal(input, aoi, output))
    else:
        _run_local(input, aoi, output, no_resume)


def _run_local(
    input_path: Path,
    aoi_path: Path | None,
    output_dir: Path,
    no_resume: bool = False,
) -> None:
    """Run the full processing pipeline locally using core services."""
    from cli.checkpoint import (
        STEP_DETECT,
        STEP_EXPORT,
        STEP_INDICATORS,
        STEP_LOAD,
        STEP_PP_CLIP,
        STEP_PP_CONFIDENCE,
        STEP_PP_NMS,
        STEP_PP_SHAPE,
        STEP_PP_SIMPLIFY,
        STEP_PP_SIZE,
        STEP_TILE,
        CheckpointManager,
        compute_fingerprint,
    )
    from core.services.detector import DetectorService
    from core.services.exporter import GISExporterService
    from core.services.imagery import ImageryLoaderService
    from core.services.indicators import IndicatorCalculatorService
    from core.services.postprocessor import PostprocessingConfig, PostprocessorService
    from core.services.tiler import TilerService

    # Initialize checkpoint manager
    fingerprint = compute_fingerprint(input_path, aoi_path, settings)
    ckpt = CheckpointManager(output_dir, fingerprint)
    if no_resume:
        ckpt.clear()
        ckpt = CheckpointManager(output_dir, fingerprint)

    # Step 1: Load imagery
    if ckpt.is_step_complete(STEP_LOAD):
        typer.echo("Loading imagery... (cached)")
        data, transform, crs, aoi_geom = ckpt.load_imagery()
    else:
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
                data, transform, crs = loader.clip_to_aoi(
                    dataset, aoi_geom, aoi_crs=str(dataset.crs)
                )
        finally:
            dataset.close()
        typer.echo(f"  Clipped shape: {data.shape}")
        ckpt.save_imagery(data, transform, crs, aoi_geom)
        ckpt.mark_complete(STEP_LOAD)

    # Step 2: Tile imagery
    if ckpt.is_step_complete(STEP_TILE):
        typer.echo("Tiling imagery... (cached)")
        tiles = ckpt.load_tiles()
        typer.echo(f"  {len(tiles)} tiles")
    else:
        typer.echo("Tiling imagery...")
        tiler = TilerService(
            tile_size=settings.tile_size, overlap=settings.tile_overlap
        )
        tiles = list(tiler.generate_tiles(data, transform, crs=crs))
        typer.echo(f"  Generated {len(tiles)} tiles")
        ckpt.save_tiles(tiles)
        ckpt.mark_complete(STEP_TILE)

    # Step 3: Detect objects
    if ckpt.is_step_complete(STEP_DETECT):
        typer.echo("Running detection... (cached)")
        all_detections = ckpt.load_detections(STEP_DETECT)
        typer.echo(f"  {len(all_detections)} raw detections")
    else:
        typer.echo("Running detection...")
        in_channels = tiles[0].data.shape[0] if tiles else 3
        detector = DetectorService(device=settings.device)
        detector.load_models(in_channels=in_channels)
        all_detections = []
        for i, tile in enumerate(tiles):
            tile_dets = detector.predict_tile(tile)
            all_detections.extend(tile_dets)
            typer.echo(f"  Tile {i + 1}/{len(tiles)}: {len(tile_dets)} detections")
        typer.echo(f"  Total raw detections: {len(all_detections)}")
        ckpt.save_detections(STEP_DETECT, all_detections, crs)
        ckpt.mark_complete(STEP_DETECT)

    # Step 4: Post-process (individual substeps)
    postprocessor = PostprocessorService()
    config = PostprocessingConfig(
        iou_threshold=settings.nms_iou_threshold,
        confidence_threshold=settings.confidence_threshold,
        min_area_m2=settings.min_area_m2,
        max_area_m2=settings.max_area_m2,
        simplify_tolerance_m=settings.simplify_tolerance_m,
    )

    # 4a: NMS
    if ckpt.is_step_complete(STEP_PP_NMS):
        typer.echo("Post-processing: NMS... (cached)")
        result = ckpt.load_detections(STEP_PP_NMS)
    else:
        typer.echo("Post-processing: NMS...")
        result = postprocessor.apply_nms(all_detections, config.iou_threshold)
        typer.echo(f"  {len(all_detections)} -> {len(result)}")
        ckpt.save_detections(STEP_PP_NMS, result, crs)
        ckpt.mark_complete(STEP_PP_NMS)

    # 4b: Confidence filter
    if ckpt.is_step_complete(STEP_PP_CONFIDENCE):
        typer.echo("Post-processing: confidence filter... (cached)")
        result = ckpt.load_detections(STEP_PP_CONFIDENCE)
    else:
        typer.echo("Post-processing: confidence filter...")
        prev_count = len(result)
        result = postprocessor.filter_by_confidence(result, config.confidence_threshold)
        typer.echo(f"  {prev_count} -> {len(result)}")
        ckpt.save_detections(STEP_PP_CONFIDENCE, result, crs)
        ckpt.mark_complete(STEP_PP_CONFIDENCE)

    # 4c: Size filter
    if ckpt.is_step_complete(STEP_PP_SIZE):
        typer.echo("Post-processing: size filter... (cached)")
        result = ckpt.load_detections(STEP_PP_SIZE)
    else:
        typer.echo("Post-processing: size filter...")
        prev_count = len(result)
        result = postprocessor.filter_by_size(
            result, config.min_area_m2, config.max_area_m2
        )
        typer.echo(f"  {prev_count} -> {len(result)}")
        ckpt.save_detections(STEP_PP_SIZE, result, crs)
        ckpt.mark_complete(STEP_PP_SIZE)

    # 4d: Shape filter
    if ckpt.is_step_complete(STEP_PP_SHAPE):
        typer.echo("Post-processing: shape filter... (cached)")
        result = ckpt.load_detections(STEP_PP_SHAPE)
    else:
        typer.echo("Post-processing: shape filter...")
        prev_count = len(result)
        result = postprocessor.filter_by_shape(result)
        typer.echo(f"  {prev_count} -> {len(result)}")
        ckpt.save_detections(STEP_PP_SHAPE, result, crs)
        ckpt.mark_complete(STEP_PP_SHAPE)

    # 4e: Simplify
    if ckpt.is_step_complete(STEP_PP_SIMPLIFY):
        typer.echo("Post-processing: simplify... (cached)")
        result = ckpt.load_detections(STEP_PP_SIMPLIFY)
    else:
        typer.echo("Post-processing: simplify...")
        prev_count = len(result)
        result = postprocessor.simplify_geometries(result, config.simplify_tolerance_m)
        typer.echo(f"  {prev_count} -> {len(result)}")
        ckpt.save_detections(STEP_PP_SIMPLIFY, result, crs)
        ckpt.mark_complete(STEP_PP_SIMPLIFY)

    # 4f: Clip to AOI
    if ckpt.is_step_complete(STEP_PP_CLIP):
        typer.echo("Post-processing: clip to AOI... (cached)")
        result = ckpt.load_detections(STEP_PP_CLIP)
    else:
        typer.echo("Post-processing: clip to AOI...")
        prev_count = len(result)
        result = postprocessor.clip_to_aoi(result, aoi_geom)
        typer.echo(f"  {prev_count} -> {len(result)}")
        ckpt.save_detections(STEP_PP_CLIP, result, crs)
        ckpt.mark_complete(STEP_PP_CLIP)

    filtered = result
    typer.echo(f"  Final: {len(filtered)} detections")

    # Step 5: Export
    if ckpt.is_step_complete(STEP_EXPORT):
        typer.echo("Exporting results... (cached)")
    else:
        typer.echo("Exporting results...")
        exporter = GISExporterService()
        paths = exporter.export_all(filtered, crs, output_dir)
        for fmt, path in paths.items():
            typer.echo(f"  {fmt}: {path}")
        ckpt.mark_complete(STEP_EXPORT)

    # Step 6: Compute indicators (use AOI as a single zone)
    if ckpt.is_step_complete(STEP_INDICATORS):
        typer.echo("Computing indicators... (cached)")
    else:
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
        ckpt.mark_complete(STEP_INDICATORS)

    typer.echo("Done!")


async def _run_temporal(
    input_path: Path, aoi_path: Path | None, output_dir: Path
) -> None:
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
