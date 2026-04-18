"""IndicatorCalculatorService -- per-zone quantitative metrics."""

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pyproj import Geod
from shapely.geometry.base import BaseGeometry

from core.services.detector import RawDetection


@dataclass
class ZoneIndicatorResult:
    """Computed indicator for a single zone and class."""

    zone_id: str
    class_name: str
    count: int
    density_per_km2: float
    total_area_m2: float


class IndicatorCalculatorService:
    """Computes per-zone quantitative metrics from detections."""

    def __init__(self) -> None:
        self._geod = Geod(ellps="WGS84")

    def compute(
        self,
        detections: list[RawDetection],
        zones: Mapping[str, BaseGeometry],
    ) -> list[ZoneIndicatorResult]:
        """Compute per-zone, per-class indicators via spatial intersection.

        Args:
            detections: List of RawDetection instances.
            zones: Dict mapping zone_id to zone geometry.

        Returns:
            List of ZoneIndicatorResult, one per zone/class combination.
        """
        results: list[ZoneIndicatorResult] = []

        for zone_id, zone_geom in zones.items():
            zone_area_m2 = abs(self._geod.geometry_area_perimeter(zone_geom)[0])
            zone_area_km2 = zone_area_m2 / 1_000_000.0 if zone_area_m2 > 0 else 1.0

            # Group detections that intersect this zone, by class
            class_stats: dict[str, list[float]] = {}
            for det in detections:
                if not det.geometry.intersects(zone_geom):
                    continue
                clipped = det.geometry.intersection(zone_geom)
                if clipped.is_empty:
                    continue
                area_m2 = abs(self._geod.geometry_area_perimeter(clipped)[0])
                class_stats.setdefault(det.class_name, []).append(area_m2)

            for class_name, areas in class_stats.items():
                count = len(areas)
                total_area = sum(areas)
                density = count / zone_area_km2
                results.append(
                    ZoneIndicatorResult(
                        zone_id=zone_id,
                        class_name=class_name,
                        count=count,
                        density_per_km2=round(density, 4),
                        total_area_m2=round(total_area, 2),
                    )
                )

        return results

    def export_csv(
        self, indicators: list[ZoneIndicatorResult], path: str | Path
    ) -> Path:
        """Export indicators to CSV.

        Args:
            indicators: List of computed indicators.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "zone_id",
            "class_name",
            "count",
            "density_per_km2",
            "total_area_m2",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ind in indicators:
                writer.writerow(
                    {
                        "zone_id": ind.zone_id,
                        "class_name": ind.class_name,
                        "count": ind.count,
                        "density_per_km2": ind.density_per_km2,
                        "total_area_m2": ind.total_area_m2,
                    }
                )
        return path

    def export_json(
        self, indicators: list[ZoneIndicatorResult], path: str | Path
    ) -> Path:
        """Export indicators to JSON.

        Args:
            indicators: List of computed indicators.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "zone_id": ind.zone_id,
                "class_name": ind.class_name,
                "count": ind.count,
                "density_per_km2": ind.density_per_km2,
                "total_area_m2": ind.total_area_m2,
            }
            for ind in indicators
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def generate_summary_table(
        self, indicators: list[ZoneIndicatorResult]
    ) -> list[dict]:
        """Generate a summary table grouped by zone.

        Args:
            indicators: List of computed indicators.

        Returns:
            List of dicts, one per zone, with per-class breakdowns.
        """
        by_zone: dict[str, dict[str, ZoneIndicatorResult]] = {}
        for ind in indicators:
            by_zone.setdefault(ind.zone_id, {})[ind.class_name] = ind

        summary: list[dict] = []
        for zone_id, classes in by_zone.items():
            row: dict = {
                "zone_id": zone_id,
                "total_detections": sum(c.count for c in classes.values()),
                "total_area_m2": round(
                    sum(c.total_area_m2 for c in classes.values()), 2
                ),
            }
            for class_name, ind in classes.items():
                row[f"{class_name}_count"] = ind.count
                row[f"{class_name}_area_m2"] = ind.total_area_m2
                row[f"{class_name}_density_per_km2"] = ind.density_per_km2
            summary.append(row)

        return summary
