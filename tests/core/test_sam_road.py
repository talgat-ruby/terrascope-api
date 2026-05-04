"""Tests for SamRoadProcess wiring + pure-numpy helpers.

We don't load the model (that needs ~3 GB of weights and a torch install
with the SAM image encoder). The wiring + post-processing helpers are pure
numpy/scipy and can be tested in isolation.
"""

from __future__ import annotations

import numpy as np

from core import ProcessSpec, build, registered_names
from core.processes.sam_road import (
    SamRoadProcess,
    _merge_close_nodes,
    _strides,
)


def test_sam_road_registered() -> None:
    assert "sam-road" in registered_names()


def test_sam_road_spacenet_registered() -> None:
    assert "sam-road-spacenet" in registered_names()


def test_spacenet_preset_applies_defaults() -> None:
    proc = build(ProcessSpec(name="sam-road-spacenet", kwargs={"device": "cpu"}))
    assert isinstance(proc, SamRoadProcess)
    assert proc.weights_filename == "spacenet_vitb_256_e10.ckpt"
    assert proc.tile_size == 1024
    assert proc.model_gsd_m_per_px == 0.3
    # The preset config dict is what feeds SAMRoad at _load() time; verify
    # the dataset-specific knobs (PATCH_SIZE + thresholds) made it through.
    assert proc._config_preset["PATCH_SIZE"] == 256
    assert proc._config_preset["ITSC_THRESHOLD"] == 0.195
    assert proc._config_preset["ROAD_THRESHOLD"] == 0.341
    assert proc._config_preset["TOPO_THRESHOLD"] == 0.705
    assert proc._config_preset["SAMPLE_MARGIN"] == 0


def test_cityscale_preset_applies_defaults() -> None:
    proc = build(ProcessSpec(name="sam-road", kwargs={"device": "cpu"}))
    assert isinstance(proc, SamRoadProcess)
    assert proc.weights_filename == "cityscale_vitb_512_e10.ckpt"
    assert proc.tile_size == 2048
    assert proc.model_gsd_m_per_px == 1.0
    assert proc._config_preset["PATCH_SIZE"] == 512


def test_spacenet_kwargs_override_preset() -> None:
    proc = build(
        ProcessSpec(
            name="sam-road-spacenet",
            kwargs={
                "tile_size": 512,
                "weights_filename": "custom.ckpt",
                "model_gsd_m_per_px": 0.5,
            },
        )
    )
    # User kwargs win; the preset only fills holes.
    assert proc.tile_size == 512
    assert proc.weights_filename == "custom.ckpt"
    assert proc.model_gsd_m_per_px == 0.5
    # Config dict still SpaceNet — the preset config is independent of kwargs.
    assert proc._config_preset["PATCH_SIZE"] == 256


def test_from_spec_defaults() -> None:
    spec = ProcessSpec(name="sam-road", kwargs={"device": "cpu"})
    proc = SamRoadProcess.from_spec(spec)
    assert proc.name == "sam-road"
    assert proc.tile_size == 2048
    assert proc.tile_overlap == 256
    assert proc.infer_patches_per_edge == 16
    assert proc.model_gsd_m_per_px == 1.0
    assert proc.source_gsd_m_per_px is None
    assert proc.weights == "congrui/sam_road"
    assert proc.weights_filename == "cityscale_vitb_512_e10.ckpt"
    assert proc.device == "cpu"


def test_scale_unset_returns_one() -> None:
    proc = SamRoadProcess.from_spec(ProcessSpec(name="sam-road"))
    assert proc._scale() == 1.0


def test_scale_resamples_high_res() -> None:
    spec = ProcessSpec(
        name="sam-road",
        kwargs={"source_gsd_m_per_px": 0.5, "model_gsd_m_per_px": 1.0},
    )
    proc = SamRoadProcess.from_spec(spec)
    # 0.5 → 1.0 means we downsample by 0.5×.
    assert proc._scale() == 0.5


def test_from_spec_overrides() -> None:
    spec = ProcessSpec(
        name="sam-road",
        kwargs={
            "tile_size": 1024,
            "tile_overlap": 128,
            "weights": "/tmp/local.ckpt",
            "min_edge_len_m": 10.0,
        },
    )
    proc = SamRoadProcess.from_spec(spec)
    assert proc.tile_size == 1024
    assert proc.tile_overlap == 128
    assert proc.weights == "/tmp/local.ckpt"
    assert proc.min_edge_len_m == 10.0


def test_strides_smaller_than_tile() -> None:
    assert _strides(1000, 2048, 256) == [0]


def test_strides_covers_full_extent() -> None:
    starts = _strides(5000, 2048, 256)
    # First tile starts at 0, every step covers (2048-256)=1792 px,
    # and the last tile is shifted to land flush with the right edge.
    assert starts[0] == 0
    assert starts[-1] + 2048 == 5000
    # Adjacent tiles overlap by exactly `overlap` (except possibly the last).
    deltas = [b - a for a, b in zip(starts, starts[1:])]
    assert all(d <= 2048 - 256 + 1 for d in deltas)


def test_merge_close_nodes_collapses_duplicates() -> None:
    # Three nodes: A and B are 5 px apart (should collapse), C is far.
    nodes = np.array([[0, 0], [4, 3], [100, 100]], dtype=np.float64)
    edges = [(0, 2), (1, 2)]
    new_nodes, new_edges = _merge_close_nodes(nodes, edges, radius=10.0)
    assert new_nodes.shape == (2, 2)
    # A+B collapsed to mean (2, 1.5 → int 1), C unchanged.
    assert {tuple(map(int, n)) for n in new_nodes} == {(2, 1), (100, 100)}
    # Both edges (originally 0→2 and 1→2) remap to the same single edge.
    assert len(new_edges) == 1


def test_merge_close_nodes_drops_self_loops() -> None:
    nodes = np.array([[0, 0], [1, 1]], dtype=np.float64)
    edges = [(0, 1)]  # within radius — collapses to a self-loop, must drop.
    _, new_edges = _merge_close_nodes(nodes, edges, radius=5.0)
    assert new_edges == []


def test_merge_close_nodes_radius_zero_is_noop() -> None:
    nodes = np.array([[0, 0], [1, 1], [10, 10]], dtype=np.float64)
    edges = [(0, 1), (1, 2)]
    new_nodes, new_edges = _merge_close_nodes(nodes, edges, radius=0)
    assert new_nodes.shape == (3, 2)
    assert new_edges == [(0, 1), (1, 2)]
