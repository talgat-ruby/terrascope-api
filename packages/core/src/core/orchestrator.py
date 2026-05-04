"""Orchestrator — run a list of Processes against one Raster, merge output.

Per-process responsibilities (download, preprocess, infer, postprocess) are
opaque to the orchestrator. After every process emits its `list[Detection]`,
the orchestrator:

1. Drops detections whose `class_name` is not in the spec's allowlist.
2. Drops detections below the spec's `min_confidence` (when the spec set
   one).
3. Stamps `source_model = process.name` for provenance.
4. Concatenates all surviving detections and assigns sequential ids 0..N
   in process order.
"""

from __future__ import annotations

from dataclasses import replace

from core.detection.types import Detection, Raster

from core.processes.base import Process


def run_processes(processes: list[Process], raster: Raster) -> list[Detection]:
    """Run each process in order and return one merged Detection list."""
    if not processes:
        raise ValueError("run_processes requires at least one process")

    merged: list[Detection] = []
    for proc in processes:
        spec = proc.spec
        allow = set(spec.classes) if spec.classes is not None else None
        min_conf = spec.min_confidence

        for det in proc.run(raster):
            if allow is not None and det.class_name not in allow:
                continue
            if min_conf is not None and det.confidence < min_conf:
                continue
            if det.source_model != proc.name:
                det = replace(det, source_model=proc.name)
            merged.append(det)

    return [replace(d, id=i) for i, d in enumerate(merged)]


__all__ = ["run_processes"]
