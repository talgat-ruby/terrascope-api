"""Load predictions / ground-truth GeoJSON for evaluation.

Predictions are expected to follow the exporter schema (`class`, `confidence`
in 0–100, geometry in EPSG:4326). Ground-truth GeoJSON only needs `class` —
confidence is taken as 1.0 when absent. Both Feature and FeatureCollection
inputs are accepted; `crs` other than WGS84 is rejected since the exporter
guarantees WGS84 output.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


def load_features(
    path: str | Path,
    *,
    class_field: str = "class",
    confidence_field: str = "confidence",
    class_filter: set[str] | None = None,
    class_remap: dict[str, str] | None = None,
) -> list[tuple[str, BaseGeometry, float]]:
    """Return `[(class_name, geom_wgs84, confidence_0_1), ...]`.

    `class_filter` keeps only the listed classes (post-remap).
    `class_remap` is applied before filtering — use it to align
    truth-side label names to prediction-side names (e.g.
    `{"car": "small vehicle"}`).
    """
    path = Path(path)
    data = json.loads(path.read_text())

    crs_name = _read_crs_name(data)
    if crs_name and not _is_wgs84(crs_name):
        raise ValueError(
            f"{path}: CRS {crs_name!r} is not WGS84 — reproject before evaluating."
        )

    feats = _features(data)
    out: list[tuple[str, BaseGeometry, float]] = []
    for f in feats:
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        props = f.get("properties") or {}
        cname = props.get(class_field)
        if cname is None:
            continue
        cname = str(cname)
        if class_remap and cname in class_remap:
            cname = class_remap[cname]
        if class_filter is not None and cname not in class_filter:
            continue

        geom = shape(geom_dict)
        if geom.is_empty:
            continue

        raw_conf = props.get(confidence_field)
        conf = _normalise_confidence(raw_conf)
        out.append((cname, geom, conf))
    return out


def _features(data: dict) -> list[dict]:
    t = data.get("type")
    if t == "FeatureCollection":
        return data.get("features") or []
    if t == "Feature":
        return [data]
    raise ValueError(f"unsupported GeoJSON type: {t!r}")


def _read_crs_name(data: dict) -> str | None:
    crs = data.get("crs")
    if not isinstance(crs, dict):
        return None
    props = crs.get("properties") or {}
    name = props.get("name")
    return str(name) if name else None


def _is_wgs84(name: str) -> bool:
    n = name.upper().replace("URN:OGC:DEF:CRS:", "")
    return n in {"EPSG:4326", "OGC:CRS84", "CRS84", "EPSG::4326"}


def _normalise_confidence(value: object) -> float:
    if value is None:
        return 1.0
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    # Exporter writes 0-100; tolerate 0-1 inputs from hand-labels too.
    return v / 100.0 if v > 1.0 else v


__all__ = ["load_features"]
