"""Process protocol + declarative spec.

A `Process` is a self-contained pipeline for one (model_family, class-set)
combination. It receives a `Raster` and emits canonical `Detection`s — the
same type `core` uses, so downstream merge/export/persistence stays uniform.

This intentionally mirrors `core.detection.Detector` rather than replacing
it; the difference is that processes own MORE than inference: they may
download external data, run preprocessing peculiar to the model
(GSD downsample, prompts, tiling), and post-process in their own way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.detection.types import Detection, Raster


@dataclass(frozen=True)
class ProcessSpec:
    """Declarative job config for one process.

    `name`            registry key (factory lookup).
    `classes`         optional output-class allowlist; None = accept all.
    `min_confidence`  optional override; processes with no concept of
                      confidence (e.g. vector footprints) ignore this.
    `kwargs`          process-specific knobs forwarded to the builder.
    """

    name: str
    classes: tuple[str, ...] | None = None
    min_confidence: float | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProcessSpec":
        if "name" not in raw or not isinstance(raw["name"], str):
            raise ValueError("ProcessSpec requires a string 'name'")
        classes = raw.get("classes")
        if classes is not None:
            if not isinstance(classes, list) or not all(
                isinstance(c, str) for c in classes
            ):
                raise ValueError("'classes' must be a list of strings")
            classes = tuple(classes)
        min_conf = raw.get("min_confidence")
        if min_conf is not None:
            min_conf = float(min_conf)
            if not 0.0 <= min_conf <= 1.0:
                raise ValueError("'min_confidence' must be between 0 and 1")
        kwargs = raw.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            raise ValueError("'kwargs' must be a dict")
        return cls(
            name=raw["name"],
            classes=classes,
            min_confidence=min_conf,
            kwargs=kwargs,
        )

    @classmethod
    def list_from_config(cls, cfg: dict[str, Any]) -> list["ProcessSpec"]:
        raw = cfg.get("processes")
        if not raw:
            raise ValueError(
                "Job config must include a non-empty 'processes' list"
            )
        if not isinstance(raw, list):
            raise ValueError("'processes' must be a list")
        return [cls.from_dict(item) for item in raw]


class Process(Protocol):
    """Self-contained per-process pipeline.

    Implementations are free to download data, cache models on instance
    state, and apply whatever preprocessing they need. The output schema is
    fixed: `list[Detection]` with `class_name`, `confidence` (use 1.0 if
    not applicable), `bbox`, `pixel_bbox`, `centroid`, `source_model`.
    """

    name: str
    spec: ProcessSpec

    def run(self, raster: Raster) -> list[Detection]: ...


__all__ = ["Detection", "Process", "ProcessSpec", "Raster"]
