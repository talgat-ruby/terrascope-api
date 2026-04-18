import numpy as np
import pytest
from rasterio.transform import from_bounds

from core.services.tiler import TilerService


@pytest.fixture
def service():
    return TilerService(tile_size=512, overlap=64)


@pytest.fixture
def sample_raster():
    """1000x1000, 3-band raster with gradient pattern."""
    height, width = 1000, 1000
    data = np.random.rand(3, height, width).astype(np.float32)
    transform = from_bounds(10.0, 49.0, 11.0, 50.0, width, height)
    return data, transform


@pytest.fixture
def small_raster():
    """256x256 raster, smaller than one tile."""
    data = np.random.rand(3, 256, 256).astype(np.float32)
    transform = from_bounds(10.0, 49.0, 10.5, 49.5, 256, 256)
    return data, transform


def test_tile_count(service, sample_raster):
    """1000x1000 with tile=512, overlap=64 -> stride=448 -> 3x3=9 tiles."""
    data, transform = sample_raster
    tiles = list(service.generate_tiles(data, transform))
    # range(0, 1000, 448) = [0, 448, 896] -> 3 per axis
    assert len(tiles) == 9


def test_single_tile_small_raster(service, small_raster):
    """256x256 raster with tile=512 should produce exactly 1 padded tile."""
    data, transform = small_raster
    tiles = list(service.generate_tiles(data, transform))
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.data.shape == (3, 512, 512)
    assert tile.index == (0, 0)


def test_overlap_correctness(service, sample_raster):
    """Adjacent tiles should share overlap pixels of identical data."""
    data, transform = sample_raster
    tiles = list(service.generate_tiles(data, transform))

    # Find tiles (0,0) and (0,1) -- horizontally adjacent
    tile_00 = next(t for t in tiles if t.index == (0, 0))
    tile_01 = next(t for t in tiles if t.index == (0, 1))

    overlap = 64
    # Right edge of tile_00 overlaps with left edge of tile_01
    right_strip = tile_00.data[:, :, -overlap:]
    left_strip = tile_01.data[:, :, :overlap]
    np.testing.assert_array_equal(right_strip, left_strip)


def test_edge_padding(service, small_raster):
    """Tile from a 256x256 raster should be padded with zeros beyond 256."""
    data, transform = small_raster
    tile = list(service.generate_tiles(data, transform))[0]

    # Padded region should be zeros
    assert np.all(tile.data[:, 256:, :] == 0)
    assert np.all(tile.data[:, :, 256:] == 0)

    # Valid mask should be False in padded region
    assert np.all(tile.valid_mask[:256, :256])
    assert not np.any(tile.valid_mask[256:, :])
    assert not np.any(tile.valid_mask[:, 256:])


def test_valid_mask_interior(service, sample_raster):
    """Interior tile (not on edge) should have all-True valid_mask."""
    data, transform = sample_raster
    tiles = list(service.generate_tiles(data, transform))

    # Tile (0,0) is interior -- it's 512x512 from a 1000x1000 raster
    tile_00 = next(t for t in tiles if t.index == (0, 0))
    assert np.all(tile_00.valid_mask)


def test_tile_transform(service, sample_raster):
    """Tile transform should map pixel (0,0) to correct geographic location."""
    data, transform = sample_raster
    tiles = list(service.generate_tiles(data, transform))

    tile_00 = next(t for t in tiles if t.index == (0, 0))
    # First tile transform should equal the raster transform
    assert tile_00.transform == transform

    tile_01 = next(t for t in tiles if t.index == (0, 1))
    stride = 448
    expected_x, expected_y = transform * (stride, 0)
    actual_x, actual_y = tile_01.transform * (0, 0)
    assert pytest.approx(actual_x) == expected_x
    assert pytest.approx(actual_y) == expected_y


def test_tile_bounds(service, sample_raster):
    """tile_bounds should return correct geographic bbox."""
    data, transform = sample_raster
    tiles = list(service.generate_tiles(data, transform))

    tile_00 = next(t for t in tiles if t.index == (0, 0))
    west, south, east, north = service.tile_bounds(tile_00)

    # First tile starts at raster origin
    expected_west, expected_north = transform * (0, 0)
    expected_east, expected_south = transform * (512, 512)

    assert pytest.approx(west) == expected_west
    assert pytest.approx(north) == expected_north
    assert pytest.approx(east) == expected_east
    assert pytest.approx(south) == expected_south


def test_tile_data_dtype(service, sample_raster):
    """All tile data should be float32."""
    data, transform = sample_raster
    for tile in service.generate_tiles(data, transform):
        assert tile.data.dtype == np.float32
