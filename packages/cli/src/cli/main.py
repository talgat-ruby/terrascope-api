"""terrascope CLI — per-process pipeline (experimental).

Loads a GeoTIFF, dispatches to a list of processes (each owns its own
preprocessing + inference + postprocessing), merges their output into a
single canonical Detection list, and writes one GeoJSON + one PNG.

Job parameters can be supplied either as flags or via a single JSON file
(`--config job.json`). When both are given, individual flags override the
matching field in the config file. See `examples/job.example.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer
from shapely.geometry import shape as shapely_shape

app = typer.Typer(
    name="terrascope", help="Terrascope cli — per-process pipeline (experimental)"
)


@dataclass
class JobConfig:
    """Resolved job parameters after merging --config + flag overrides.

    Mirrors the JSON schema accepted on disk (see examples/job.example.json):

        {
          "input":  "inputs/Astana_1.tif",
          "aoi":    "inputs/Astana_aoi.geojson",
          "output": "outputs/astana",
          "processes": [
            {"name": "msft-buildings"},
            {"name": "yolov8-satellite-vehicle",
             "classes": ["car"], "min_confidence": 0.25,
             "kwargs": {"source_gsd_m_per_px": 0.054,
                        "model_gsd_m_per_px": 0.15}}
          ]
        }
    """

    input: Path
    output: Path
    processes: list[dict]
    aoi: Path | None = None


@app.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a JSON job config (input/aoi/output/processes).",
    ),
    input: Path | None = typer.Option(
        None, "--input", "-i", help="Path to GeoTIFF (overrides config)."
    ),
    aoi: Path | None = typer.Option(
        None, "--aoi", "-a", help="GeoJSON AOI (overrides config)."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (overrides config)."
    ),
    processes: str | None = typer.Option(
        None,
        "--processes",
        "-p",
        help="JSON list of process specs (overrides config).",
    ),
    list_processes: bool = typer.Option(
        False, "--list-processes", help="List registered process names and exit."
    ),
) -> None:
    if list_processes:
        from core import registered_names

        for name in registered_names():
            typer.echo(name)
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    job = _resolve_job(
        config_path=config,
        input_override=input,
        aoi_override=aoi,
        output_override=output,
        processes_override=processes,
    )
    _run(job)


def _resolve_job(
    *,
    config_path: Path | None,
    input_override: Path | None,
    aoi_override: Path | None,
    output_override: Path | None,
    processes_override: str | None,
) -> JobConfig:
    """Merge a JSON config file with per-flag overrides into a JobConfig.

    Flag values always win over the config file. Paths in the config file
    are resolved relative to the config file's directory, so a job JSON
    can be moved alongside its inputs without rewriting paths.
    """
    raw: dict = {}
    base_dir: Path | None = None
    if config_path is not None:
        if not config_path.exists():
            typer.echo(f"Error: config file not found: {config_path}", err=True)
            raise typer.Exit(code=2)
        raw = json.loads(config_path.read_text())
        if not isinstance(raw, dict):
            typer.echo("Error: config file must be a JSON object.", err=True)
            raise typer.Exit(code=2)
        base_dir = config_path.parent

    def _resolve_path(value: str | Path) -> Path:
        p = Path(value)
        if p.is_absolute() or base_dir is None:
            return p
        return (base_dir / p).resolve()

    input_value: Path | None = input_override
    if input_value is None and "input" in raw:
        input_value = _resolve_path(raw["input"])
    if input_value is None:
        typer.echo("Error: --input or config.input is required.", err=True)
        raise typer.Exit(code=2)

    aoi_value: Path | None = aoi_override
    if aoi_value is None and raw.get("aoi"):
        aoi_value = _resolve_path(raw["aoi"])

    output_value: Path | None = output_override
    if output_value is None and "output" in raw:
        output_value = _resolve_path(raw["output"])
    if output_value is None:
        output_value = Path("./output2")

    process_specs: list[dict]
    if processes_override is not None:
        process_specs = json.loads(processes_override)
    elif "processes" in raw:
        process_specs = raw["processes"]
    else:
        typer.echo(
            "Error: --processes or config.processes is required.", err=True
        )
        raise typer.Exit(code=2)
    if not isinstance(process_specs, list):
        typer.echo("Error: 'processes' must be a JSON list.", err=True)
        raise typer.Exit(code=2)

    return JobConfig(
        input=input_value,
        aoi=aoi_value,
        output=output_value,
        processes=process_specs,
    )


def _run(job: JobConfig) -> None:
    from core.services.exporter import GISExporterService
    from core.services.imagery import ImageryLoaderService

    from core import ProcessSpec, build, render_overlay, run_processes

    job.output.mkdir(parents=True, exist_ok=True)

    aoi_geom = (
        shapely_shape(json.loads(job.aoi.read_text()))
        if job.aoi is not None
        else None
    )

    typer.echo(f"Loading imagery from {job.input}...")
    raster = ImageryLoaderService().load_clipped(job.input, aoi_geom)
    typer.echo(f"  Shape: {raster.data.shape}, CRS: {raster.crs}")

    specs = [ProcessSpec.from_dict(p) for p in job.processes]
    procs = [build(s) for s in specs]
    typer.echo(f"Running {len(procs)} process(es): {[p.name for p in procs]}")

    detections = run_processes(procs, raster)
    typer.echo(f"  Merged detections: {len(detections)}")

    overlay_path = job.output / "overlay.png"
    render_overlay(raster, detections, overlay_path)
    typer.echo(f"  PNG overlay: {overlay_path}")

    geojson_path = job.output / "detections.geojson"
    GISExporterService().export_geojson(detections, geojson_path, crs=raster.crs)
    typer.echo(f"  GeoJSON: {geojson_path}")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
