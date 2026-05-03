"""Tests for cli2 JSON config loading + flag override behavior.

We test the pure resolver (`_resolve_job`) directly so the tests don't need
to load real GeoTIFFs or run any processes.
"""

import json
from pathlib import Path

import pytest
import typer

from cli2.main import _resolve_job


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_resolves_paths_relative_to_config_file(tmp_path: Path):
    cfg_dir = tmp_path / "jobs"
    cfg_dir.mkdir()
    cfg = _write(
        cfg_dir / "job.json",
        {
            "input": "../inputs/raster.tif",
            "output": "../outputs/run1",
            "processes": [{"name": "msft-buildings"}],
        },
    )
    job = _resolve_job(
        config_path=cfg,
        input_override=None,
        aoi_override=None,
        output_override=None,
        processes_override=None,
    )
    assert job.input == (tmp_path / "inputs" / "raster.tif").resolve()
    assert job.output == (tmp_path / "outputs" / "run1").resolve()
    assert job.aoi is None
    assert job.processes == [{"name": "msft-buildings"}]


def test_flag_overrides_beat_config_file(tmp_path: Path):
    cfg = _write(
        tmp_path / "job.json",
        {
            "input": "from-config.tif",
            "output": "from-config-out",
            "processes": [{"name": "msft-buildings"}],
        },
    )
    job = _resolve_job(
        config_path=cfg,
        input_override=Path("/abs/override.tif"),
        aoi_override=None,
        output_override=Path("/abs/override-out"),
        processes_override=json.dumps(
            [{"name": "yolov8-satellite-vehicle", "classes": ["car"]}]
        ),
    )
    assert job.input == Path("/abs/override.tif")
    assert job.output == Path("/abs/override-out")
    assert job.processes[0]["name"] == "yolov8-satellite-vehicle"


def test_aoi_field_in_config(tmp_path: Path):
    cfg = _write(
        tmp_path / "job.json",
        {
            "input": "raster.tif",
            "aoi": "aoi.geojson",
            "processes": [{"name": "msft-buildings"}],
        },
    )
    job = _resolve_job(
        config_path=cfg,
        input_override=None,
        aoi_override=None,
        output_override=None,
        processes_override=None,
    )
    assert job.aoi == (tmp_path / "aoi.geojson").resolve()


def test_null_aoi_in_config_is_ignored(tmp_path: Path):
    cfg = _write(
        tmp_path / "job.json",
        {
            "input": "raster.tif",
            "aoi": None,
            "processes": [{"name": "msft-buildings"}],
        },
    )
    job = _resolve_job(
        config_path=cfg,
        input_override=None,
        aoi_override=None,
        output_override=None,
        processes_override=None,
    )
    assert job.aoi is None


def test_missing_input_exits(tmp_path: Path):
    cfg = _write(
        tmp_path / "job.json",
        {"processes": [{"name": "msft-buildings"}]},
    )
    with pytest.raises(typer.Exit):
        _resolve_job(
            config_path=cfg,
            input_override=None,
            aoi_override=None,
            output_override=None,
            processes_override=None,
        )


def test_missing_processes_exits(tmp_path: Path):
    cfg = _write(tmp_path / "job.json", {"input": "raster.tif"})
    with pytest.raises(typer.Exit):
        _resolve_job(
            config_path=cfg,
            input_override=None,
            aoi_override=None,
            output_override=None,
            processes_override=None,
        )


def test_no_config_file_with_only_flags():
    job = _resolve_job(
        config_path=None,
        input_override=Path("/in.tif"),
        aoi_override=None,
        output_override=Path("/out"),
        processes_override=json.dumps([{"name": "msft-buildings"}]),
    )
    assert job.input == Path("/in.tif")
    assert job.output == Path("/out")
    assert job.processes == [{"name": "msft-buildings"}]


def test_default_output_when_unspecified():
    job = _resolve_job(
        config_path=None,
        input_override=Path("/in.tif"),
        aoi_override=None,
        output_override=None,
        processes_override=json.dumps([{"name": "msft-buildings"}]),
    )
    assert job.output == Path("./output2")


def test_example_job_file_parses():
    """Ensure the shipped sample stays valid as the schema evolves."""
    example = (
        Path(__file__).resolve().parents[2]
        / "packages" / "cli2" / "examples" / "job.example.json"
    )
    payload = json.loads(example.read_text())
    assert "input" in payload
    assert isinstance(payload["processes"], list) and payload["processes"]
    for spec in payload["processes"]:
        assert "name" in spec
