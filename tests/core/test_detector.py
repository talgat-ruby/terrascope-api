import numpy as np
import pytest
from rasterio.transform import from_bounds
from core.models.tile import Tile
from core.services.detector import DetectorService, RawDetection


@pytest.fixture
def service():
    return DetectorService(device="cpu")


@pytest.fixture
def sample_tile():
    """512x512 tile with known transform."""
    data = np.random.rand(3, 512, 512).astype(np.float32)
    transform = from_bounds(10.0, 49.0, 10.1, 49.1, 512, 512)
    valid_mask = np.ones((512, 512), dtype=np.bool_)
    return Tile(
        index=(0, 0),
        pixel_window=(0, 0, 512, 512),
        transform=transform,
        data=data,
        valid_mask=valid_mask,
    )


def test_mask_to_polygons_produces_detections(service, sample_tile):
    """A probability mask with a clear region should produce polygon detections."""
    prob_mask = np.zeros((512, 512), dtype=np.float32)
    # Create a 100x100 block of high probability
    prob_mask[100:200, 100:200] = 0.9

    detections = service._mask_to_polygons(
        prob_mask, sample_tile.transform, "building", source="test"
    )
    assert len(detections) >= 1
    assert all(isinstance(d, RawDetection) for d in detections)
    assert all(d.class_name == "building" for d in detections)
    assert all(d.source == "test" for d in detections)
    assert all(d.geometry.geom_type == "Polygon" for d in detections)


def test_mask_to_polygons_empty_mask(service, sample_tile):
    """An all-zero mask should produce no detections."""
    prob_mask = np.zeros((512, 512), dtype=np.float32)
    detections = service._mask_to_polygons(
        prob_mask, sample_tile.transform, "vegetation", source="test"
    )
    assert len(detections) == 0


def test_mask_to_polygons_below_threshold(service, sample_tile):
    """Probabilities below threshold (0.5) should produce no detections."""
    prob_mask = np.full((512, 512), 0.3, dtype=np.float32)
    detections = service._mask_to_polygons(
        prob_mask, sample_tile.transform, "road", source="test"
    )
    assert len(detections) == 0


def test_confidence_in_valid_range(service, sample_tile):
    """Confidence scores should be between 0 and 100."""
    prob_mask = np.zeros((512, 512), dtype=np.float32)
    prob_mask[50:150, 50:150] = 0.75

    detections = service._mask_to_polygons(
        prob_mask, sample_tile.transform, "water", source="test"
    )
    for d in detections:
        assert 0 <= d.confidence <= 100


def test_confidence_reflects_probability(service, sample_tile):
    """Higher probability regions should yield higher confidence."""
    # High prob region
    high_mask = np.zeros((512, 512), dtype=np.float32)
    high_mask[100:200, 100:200] = 0.95

    # Low prob region (just above threshold)
    low_mask = np.zeros((512, 512), dtype=np.float32)
    low_mask[100:200, 100:200] = 0.55

    high_dets = service._mask_to_polygons(
        high_mask, sample_tile.transform, "building", source="test"
    )
    low_dets = service._mask_to_polygons(
        low_mask, sample_tile.transform, "building", source="test"
    )
    assert len(high_dets) >= 1
    assert len(low_dets) >= 1
    assert high_dets[0].confidence > low_dets[0].confidence


def test_predict_tile_with_masks(service, sample_tile):
    """predict_tile_with_masks should convert class masks to detections."""
    masks = {
        "building": np.zeros((512, 512), dtype=np.float32),
        "vegetation": np.zeros((512, 512), dtype=np.float32),
    }
    # Add a building blob
    masks["building"][200:300, 200:300] = 0.85
    # Add a vegetation blob
    masks["vegetation"][50:100, 300:400] = 0.7

    detections = service.predict_tile_with_masks(sample_tile, masks, source="mock")
    class_names = {d.class_name for d in detections}
    assert "building" in class_names
    assert "vegetation" in class_names
    assert all(d.source == "mock" for d in detections)


def test_predict_tile_with_masks_respects_valid_mask(service):
    """Padded regions (valid_mask=False) should not produce detections."""
    data = np.random.rand(3, 512, 512).astype(np.float32)
    transform = from_bounds(10.0, 49.0, 10.1, 49.1, 512, 512)

    # Only top-left 256x256 is valid
    valid_mask = np.zeros((512, 512), dtype=np.bool_)
    valid_mask[:256, :256] = True

    tile = Tile(
        index=(0, 0),
        pixel_window=(0, 0, 256, 256),
        transform=transform,
        data=data,
        valid_mask=valid_mask,
    )

    masks = {"building": np.zeros((512, 512), dtype=np.float32)}
    # Put high prob in padded region only
    masks["building"][300:400, 300:400] = 0.9

    detections = service.predict_tile_with_masks(tile, masks, source="mock")
    assert len(detections) == 0


def test_predict_tile_with_masks_ignores_unknown_class(service, sample_tile):
    """Unknown class names not in CLASS_REGISTRY should be ignored."""
    masks = {"alien_structure": np.ones((512, 512), dtype=np.float32) * 0.9}
    detections = service.predict_tile_with_masks(sample_tile, masks, source="mock")
    assert len(detections) == 0


def test_multiple_regions_produce_multiple_detections(service, sample_tile):
    """Separate regions in a mask should produce separate detections."""
    prob_mask = np.zeros((512, 512), dtype=np.float32)
    # Two separate blocks
    prob_mask[50:100, 50:100] = 0.8
    prob_mask[300:350, 300:350] = 0.8

    detections = service._mask_to_polygons(
        prob_mask, sample_tile.transform, "building", source="test"
    )
    assert len(detections) == 2
