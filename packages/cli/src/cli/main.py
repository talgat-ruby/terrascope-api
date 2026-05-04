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
from shapely.errors import GEOSException
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry

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


def _fail(message: str, code: int = 2) -> typer.Exit:
    """Print `message` to stderr and return a typer.Exit to raise."""
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=code)


def _load_json(path: Path, what: str) -> object:
    """Read+parse JSON from `path`, surfacing decode errors as CLI errors."""
    try:
        text = path.read_text()
    except OSError as e:
        raise _fail(f"could not read {what} {path}: {e}") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise _fail(
            f"{what} {path} is not valid JSON (line {e.lineno}, col {e.colno}): {e.msg}"
        ) from e


def _parse_json_str(text: str, what: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise _fail(
            f"{what} is not valid JSON (line {e.lineno}, col {e.colno}): {e.msg}"
        ) from e


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
            raise _fail(f"config file not found: {config_path}")
        parsed = _load_json(config_path, "config file")
        if not isinstance(parsed, dict):
            raise _fail("config file must be a JSON object.")
        raw = parsed
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
        raise _fail("--input or config.input is required.")

    aoi_value: Path | None = aoi_override
    if aoi_value is None and raw.get("aoi"):
        aoi_value = _resolve_path(raw["aoi"])

    output_value: Path | None = output_override
    if output_value is None and "output" in raw:
        output_value = _resolve_path(raw["output"])
    if output_value is None:
        output_value = Path("./output")

    process_specs: list[dict]
    if processes_override is not None:
        parsed_p = _parse_json_str(processes_override, "--processes")
        if not isinstance(parsed_p, list):
            raise _fail("--processes must be a JSON list.")
        process_specs = parsed_p
    elif "processes" in raw:
        if not isinstance(raw["processes"], list):
            raise _fail("'processes' must be a list.")
        process_specs = raw["processes"]
    else:
        raise _fail("--processes or config.processes is required.")

    return JobConfig(
        input=input_value,
        aoi=aoi_value,
        output=output_value,
        processes=process_specs,
    )


def _load_aoi(path: Path) -> BaseGeometry:
    """Read a GeoJSON AOI from disk, surfacing decode/geometry errors clearly."""
    if not path.exists():
        raise _fail(f"AOI file not found: {path}")
    parsed = _load_json(path, "AOI file")
    if not isinstance(parsed, dict):
        raise _fail(f"AOI file {path} must contain a GeoJSON object.")
    # Accept either a bare geometry or a Feature/FeatureCollection.
    geom_dict = parsed
    if parsed.get("type") == "Feature":
        geom_dict = parsed.get("geometry") or {}
    elif parsed.get("type") == "FeatureCollection":
        feats = parsed.get("features") or []
        if not feats:
            raise _fail(f"AOI file {path} has no features.")
        geom_dict = feats[0].get("geometry") or {}
    try:
        return shapely_shape(geom_dict)
    except (GEOSException, ValueError, TypeError, KeyError) as e:
        raise _fail(f"AOI file {path} is not a valid geometry: {e}") from e


def _run(job: JobConfig) -> None:
    from core import (
        ProcessSpec,
        build,
        export_geojson,
        load_raster,
        render_overlay,
        run_processes,
        write_indicators,
    )

    job.output.mkdir(parents=True, exist_ok=True)

    aoi_geom = _load_aoi(job.aoi) if job.aoi is not None else None

    if not job.input.exists():
        raise _fail(f"input GeoTIFF not found: {job.input}")

    typer.echo(f"Loading imagery from {job.input}...")
    raster = load_raster(job.input, aoi_geom)
    typer.echo(f"  Shape: {raster.data.shape}, CRS: {raster.crs}")

    try:
        specs = [ProcessSpec.from_dict(p) for p in job.processes]
        procs = [build(s) for s in specs]
    except ValueError as e:
        raise _fail(str(e)) from e
    typer.echo(f"Running {len(procs)} process(es): {[p.name for p in procs]}")

    detections = run_processes(procs, raster)
    typer.echo(f"  Merged detections: {len(detections)}")

    overlay_path = job.output / "overlay.png"
    render_overlay(raster, detections, overlay_path)
    typer.echo(f"  PNG overlay: {overlay_path}")

    # Scene identifier for the §5 `source` field — input filename stem.
    scene_id = job.input.stem
    geojson_path = job.output / "detections.geojson"
    export_geojson(detections, geojson_path, crs=raster.crs, source=scene_id)
    typer.echo(f"  GeoJSON (EPSG:4326): {geojson_path}")

    indicators_json, indicators_csv = write_indicators(
        detections, job.output, raster=raster
    )
    typer.echo(f"  Indicators:  {indicators_json}, {indicators_csv}")

    typer.echo("Done.")


@app.command("eval")
def eval_cmd(
    pred: Path = typer.Option(
        ..., "--pred", "-p", help="Predictions GeoJSON (exporter output)."
    ),
    truth: Path = typer.Option(
        ..., "--truth", "-t", help="Ground-truth GeoJSON (must be EPSG:4326)."
    ),
    output: Path = typer.Option(
        Path("./eval"), "--output", "-o", help="Directory for metrics.json / .csv."
    ),
    iou: float = typer.Option(0.5, "--iou", help="IoU threshold for TP match."),
    kind: str = typer.Option(
        "polygon", "--kind", help="IoU kind: 'polygon' or 'line' (buffered)."
    ),
    line_buffer_m: float = typer.Option(
        5.0, "--line-buffer-m", help="Buffer (m) for line-IoU; ignored for polygon."
    ),
    classes: str | None = typer.Option(
        None,
        "--classes",
        help="Comma-separated class allowlist (post-remap). Default: all.",
    ),
    truth_class_field: str = typer.Option(
        "class", "--truth-class-field", help="Property name holding class on truth."
    ),
    truth_remap: str | None = typer.Option(
        None,
        "--truth-remap",
        help='JSON dict remapping truth labels to prediction labels, e.g. \'{"car":"small vehicle"}\'.',
    ),
) -> None:
    """Score predictions GeoJSON against a ground-truth GeoJSON (§7).

    Both inputs must be in EPSG:4326. The predictions file is expected to
    follow the exporter schema (`class`, `confidence` in 0–100). Truth files
    only need `class`; missing confidence is treated as 1.0.
    """
    from core import evaluate, load_features

    class_filter: set[str] | None = None
    if classes:
        class_filter = {c.strip() for c in classes.split(",") if c.strip()}

    remap: dict[str, str] | None = None
    if truth_remap:
        parsed = _parse_json_str(truth_remap, "--truth-remap")
        if not isinstance(parsed, dict):
            raise _fail("--truth-remap must be a JSON object.")
        remap = {str(k): str(v) for k, v in parsed.items()}

    if not pred.exists():
        raise _fail(f"predictions file not found: {pred}")
    if not truth.exists():
        raise _fail(f"truth file not found: {truth}")

    pred_records = load_features(pred, class_filter=class_filter)
    truth_records = load_features(
        truth,
        class_field=truth_class_field,
        class_filter=class_filter,
        class_remap=remap,
    )

    preds_by_class: dict[str, list[tuple[object, float]]] = {}
    for cname, geom, conf in pred_records:
        preds_by_class.setdefault(cname, []).append((geom, conf))
    truths_by_class: dict[str, list[object]] = {}
    for cname, geom, _ in truth_records:
        truths_by_class.setdefault(cname, []).append(geom)

    report = evaluate(
        preds_by_class,  # type: ignore[arg-type]
        truths_by_class,  # type: ignore[arg-type]
        iou_threshold=iou,
        iou_kind=kind,
        line_buffer_m=line_buffer_m,
    )

    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "metrics.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))

    csv_path = output / "metrics.csv"
    import csv as _csv

    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            [
                "class",
                "iou_threshold",
                "n_pred",
                "n_truth",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "average_precision",
            ]
        )
        for c in report.by_class:
            w.writerow(
                [
                    c.class_name,
                    c.iou_threshold,
                    c.n_pred,
                    c.n_truth,
                    c.tp,
                    c.fp,
                    c.fn,
                    f"{c.precision:.4f}",
                    f"{c.recall:.4f}",
                    f"{c.f1:.4f}",
                    f"{c.average_precision:.4f}",
                ]
            )

    typer.echo(f"Wrote {json_path} and {csv_path}")
    typer.echo(
        f"micro P/R/F1: {report.micro_precision:.3f} / "
        f"{report.micro_recall:.3f} / {report.micro_f1:.3f}  |  "
        f"mAP@{iou}: {report.mean_average_precision:.3f}"
    )
    for c in report.by_class:
        typer.echo(
            f"  {c.class_name:<20s} P={c.precision:.3f} R={c.recall:.3f} "
            f"F1={c.f1:.3f} AP={c.average_precision:.3f} "
            f"(TP={c.tp} FP={c.fp} FN={c.fn})"
        )


if __name__ == "__main__":
    app()
