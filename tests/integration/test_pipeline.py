"""End-to-end local pipeline test with a stub Detector."""

from pathlib import Path

import geopandas as gpd

from core.detection import filter_detections, render_overlay
from core.detection.types import Detection, Raster
from core.services.exporter import GISExporterService
from core.services.imagery import ImageryLoaderService


class _StubDetector:
    name = "stub"

    def detect(self, raster: Raster) -> list[Detection]:
        # Two synthetic detections sized to fit inside the synthetic AOI.
        from shapely.geometry import box

        b1 = (10.02, 49.92, 10.03, 49.93)
        b2 = (10.05, 49.95, 10.06, 49.96)
        return [
            Detection(
                id=0,
                class_name="car",
                confidence=0.9,
                bbox=b1,
                pixel_bbox=(10, 10, 30, 30),
                centroid=box(*b1).centroid,
            ),
            Detection(
                id=1,
                class_name="building",
                confidence=0.7,
                bbox=b2,
                pixel_bbox=(40, 40, 60, 60),
                centroid=box(*b2).centroid,
            ),
        ]


def test_full_local_pipeline(synthetic_geotiff: Path, sample_aoi, tmp_path: Path):
    raster = ImageryLoaderService().load_clipped(synthetic_geotiff, sample_aoi)
    assert raster.data.shape[2] == 3  # HWC

    detections = _StubDetector().detect(raster)
    detections = filter_detections(detections, min_confidence=0.5, aoi=sample_aoi)
    assert len(detections) == 2

    overlay = render_overlay(raster, detections, tmp_path / "overlay.png")
    assert overlay.exists()
    assert overlay.stat().st_size > 0

    geojson = GISExporterService().export_geojson(
        detections, tmp_path / "detections.geojson", crs=raster.crs
    )
    gdf = gpd.read_file(geojson)
    assert len(gdf) == 2
    assert "class_name" in gdf.columns
    assert "centroid_wkt" in gdf.columns
    assert all(g.geom_type == "Polygon" for g in gdf.geometry)


def test_pipeline_empty_detections(synthetic_geotiff: Path, sample_aoi, tmp_path: Path):
    raster = ImageryLoaderService().load_clipped(synthetic_geotiff, sample_aoi)
    detections = filter_detections([], min_confidence=0.5, aoi=sample_aoi)
    assert detections == []

    overlay = render_overlay(raster, detections, tmp_path / "overlay.png")
    assert overlay.exists()

    geojson = GISExporterService().export_geojson(
        detections, tmp_path / "empty.geojson", crs=raster.crs
    )
    gdf = gpd.read_file(geojson)
    assert len(gdf) == 0
