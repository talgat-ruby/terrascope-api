"""DetectorSpec — declarative config for one detector inside a job.

A job declares a list of these; the factory builds a (possibly composite)
`Detector` from them. Each spec scopes a detector to a class allowlist and
optionally overrides its confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    classes: tuple[str, ...] | None = None
    min_confidence: float | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DetectorSpec":
        if "name" not in raw or not isinstance(raw["name"], str):
            raise ValueError("DetectorSpec requires a string 'name'")
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
    def list_from_config(cls, cfg: dict[str, Any]) -> list["DetectorSpec"]:
        raw = cfg.get("detectors")
        if not raw:
            raise ValueError(
                "Job config must include a non-empty 'detectors' list"
            )
        if not isinstance(raw, list):
            raise ValueError("'detectors' must be a list")
        return [cls.from_dict(item) for item in raw]
