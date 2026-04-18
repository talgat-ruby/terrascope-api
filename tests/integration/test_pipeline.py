"""End-to-end local pipeline test with mocked detector."""

from pathlib import Path

import geopandas as gpd
import numpy as np

from core.services.detector import DetectorService, RawDetection
from core.services.exporter import GISExporterService
from core.services.imagery import ImageryLoaderService
from core.services.indicators import IndicatorCalculatorService
from core.services.postprocessor import PostprocessingConfig, PostprocessorService
from core.services.tiler import TilerService


def test_full_local_pipeline(synthetic_geotiff: Path, sample_aoi, mock_detector: DetectorService, tmp_path: Path):
    """Run load → tile → detect → postprocess → export → indicators."""
    # Step 1: Load imagery
    loader = ImageryLoaderService()
    dataset = loader.load(str(synthetic_geotiff))
    try:
        data, transform, crs = loader.clip_to_aoi(dataset, sample_aoi)
    finally:
        dataset.close()

    assert data.shape[0] == 3
    assert crs == "EPSG:4326"

    # Step 2: Tile
    tiler = TilerService(tile_size=64, overlap=8)
    tiles = list(tiler.generate_tiles(data, transform, crs=crs))
    assert len(tiles) >= 1

    # Step 3: Detect (mock detector — exercises real _mask_to_polygons)
    all_detections: list[RawDetection] = []
    for tile in tiles:
        all_detections.extend(mock_detector.predict_tile(tile))

    assert len(all_detections) > 0
    class_names = {d.class_name for d in all_detections}
    assert "building" in class_names or "vegetation" in class_names

    # Step 4: Post-process (relaxed thresholds)
    postprocessor = PostprocessorService()
    config = PostprocessingConfig(
        iou_threshold=0.5,
        confidence_threshold=30,
        min_area_m2=0.0,
        max_area_m2=1_000_000_000.0,
        simplify_tolerance_m=0.0,
    )
    filtered, stats = postprocessor.run(all_detections, config, sample_aoi)

    assert stats["input_count"] == len(all_detections)
    assert stats["output_count"] <= stats["input_count"]
    assert len(filtered) > 0

    # Step 5: Export
    export_dir = tmp_path / "export"
    exporter = GISExporterService()
    paths = exporter.export_all(filtered, crs, export_dir)

    assert "geojson" in paths
    assert "gpkg" in paths
    assert "shp" in paths
    for fmt_path in paths.values():
        assert Path(fmt_path).exists()

    # Read back GeoJSON and validate
    gdf = gpd.read_file(paths["geojson"])
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == len(filtered)
    assert "class_name" in gdf.columns
    assert "confidence" in gdf.columns
    assert "area_m2" in gdf.columns
    assert gdf.geometry.is_valid.all()

    # Step 6: Indicators
    calculator = IndicatorCalculatorService()
    indicators = calculator.compute(filtered, {"test_zone": sample_aoi})

    assert len(indicators) > 0
    for ind in indicators:
        assert ind.zone_id == "test_zone"
        assert ind.count > 0
        assert ind.density_per_km2 > 0
        assert ind.total_area_m2 > 0


def test_pipeline_empty_detections(synthetic_geotiff: Path, sample_aoi, tmp_path: Path):
    """Pipeline completes gracefully when detector finds nothing."""
    # Load & tile
    loader = ImageryLoaderService()
    dataset = loader.load(str(synthetic_geotiff))
    try:
        data, transform, crs = loader.clip_to_aoi(dataset, sample_aoi)
    finally:
        dataset.close()

    tiler = TilerService(tile_size=64, overlap=8)
    tiles = list(tiler.generate_tiles(data, transform, crs=crs))

    # Detector with empty masks → no detections
    detector = DetectorService(device="cpu")
    all_detections: list[RawDetection] = []
    for tile in tiles:
        h, w = tile.data.shape[1], tile.data.shape[2]
        empty_masks: dict[str, np.ndarray] = {
            "building": np.zeros((h, w), dtype=np.float32),
        }
        all_detections.extend(
            detector.predict_tile_with_masks(tile, empty_masks, source="mock")
        )

    assert len(all_detections) == 0

    # Post-process empty list
    postprocessor = PostprocessorService()
    filtered, stats = postprocessor.run(all_detections, PostprocessingConfig())
    assert len(filtered) == 0
    assert stats["output_count"] == 0

    # Export empty
    export_dir = tmp_path / "export_empty"
    exporter = GISExporterService()
    paths = exporter.export_all(filtered, crs, export_dir)
    gdf = gpd.read_file(paths["geojson"])
    assert len(gdf) == 0

    # Indicators on empty
    calculator = IndicatorCalculatorService()
    indicators = calculator.compute(filtered, {"zone": sample_aoi})
    assert len(indicators) == 0
