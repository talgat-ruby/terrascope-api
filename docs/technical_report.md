# Terrascope -- Technical Report

## 1. Introduction

### 1.1 Problem Statement

Automated extraction of objects and spatial features from satellite imagery is essential for territory monitoring, infrastructure inventory, and quantitative analysis. Manual annotation is prohibitively slow for large areas, while modern deep learning models enable scalable detection with coordinate-referenced output.

### 1.2 Solution Scope

Terrascope is a prototype system that:

- Processes GeoTIFF satellite imagery within a defined area of interest (AOI)
- Detects four object classes: **buildings**, **roads**, **vegetation**, and **water bodies**
- Exports georeferenced results as GIS layers (GeoJSON, GeoPackage, Shapefile)
- Computes zone-level quantitative indicators (count, density, area)
- Provides quality evaluation metrics (Precision, Recall, F1, IoU, mAP)

### 1.3 Architecture Overview

The system is built as a Python 3.14 monorepo with four packages:

| Package  | Role                                      |
|----------|-------------------------------------------|
| `core`   | Models, schemas, services, configuration  |
| `api`    | FastAPI REST endpoints                    |
| `worker` | Temporal workflow activities              |
| `cli`    | Typer command-line interface              |

Key infrastructure: PostgreSQL + PostGIS for storage, Temporal for workflow orchestration, Docker for containerization.

---

## 2. Data Description

### 2.1 Input Data

| Input              | Format                     | Requirements                                |
|--------------------|----------------------------|---------------------------------------------|
| Satellite imagery  | GeoTIFF (3-band RGB)       | Must have CRS and geotransform              |
| Area of interest   | GeoJSON polygon            | Any CRS (auto-reprojected if needed)        |

The system validates CRS presence at load time and raises an error for non-georeferenced files. If the AOI is in a different CRS than the imagery, automatic reprojection is performed via `pyproj.Transformer`.

### 2.2 Coordinate Reference System

- Input CRS is preserved through the processing pipeline via explicit `crs` parameter propagation
- Every `Tile` object carries its CRS string for downstream operations
- GeoJSON export always reprojects to **WGS84 (EPSG:4326)** per RFC 7946
- PostGIS storage uses **SRID=4326** via GeoAlchemy2
- All area and distance computations use `pyproj.Geod(ellps="WGS84")` for geodesic accuracy

### 2.3 Output Fields

Each detection record in the exported layer contains:

| Field        | Type      | Description                               |
|--------------|-----------|-------------------------------------------|
| id           | integer   | Unique identifier (auto-generated)        |
| class_name   | string    | Detection class (building, road, etc.)    |
| confidence   | integer   | Confidence score (0--100)                 |
| source       | string    | Model identifier (torchgeo, samgeo)       |
| geometry     | Polygon   | Georeferenced detection boundary          |
| area_m2      | float     | Geodesic area in square meters            |

---

## 3. Approach

### 3.1 Processing Pipeline

The system implements a 6-step pipeline, available both as a local CLI command and as a Temporal workflow:

```
Load Imagery -> Tile -> Detect Objects -> Post-process -> Export -> Compute Indicators
```

1. **Load imagery:** Validate CRS, clip raster to AOI using `rasterio.mask.mask(crop=True)`
2. **Tile:** Split into overlapping tiles (default 512x512, overlap 64px) with zero-padding at edges; `valid_mask` tracks real vs padded pixels
3. **Detect objects:** Run ML inference per tile, convert probability masks to vector polygons
4. **Post-process:** NMS, filtering (confidence, size, shape), simplification, AOI clipping
5. **Export:** Write GeoJSON, GeoPackage, and Shapefile with computed area_m2
6. **Indicators:** Compute per-zone count, density, and total area

### 3.2 Tiling Strategy

Large imagery is split into tiles with configurable `tile_size` and `overlap`:

- **Stride:** `tile_size - overlap` ensures adjacent tiles overlap to avoid missing detections at boundaries
- **Edge padding:** Tiles at raster edges are zero-padded to maintain uniform dimensions; `valid_mask=False` for padded pixels prevents false detections
- **Tile metadata:** Each tile stores its pixel window, affine transform, and CRS for accurate georeferencing

### 3.3 Detection Models

#### TorchGeo FCN (Vegetation, Road, Water)

- Fully Convolutional Network from the `torchgeo` library
- Semantic segmentation: produces per-class probability masks
- Architecture: FCN with background channel (index 0) + 3 class channels
- Input: 3-band tile tensor; Output: `dict[str, NDArray[float32]]`

#### SAMGeo -- Segment Anything Model (Buildings)

- Instance segmentation via `segment-geospatial`
- Zero-shot capability: no task-specific training required
- Interface: writes tile to temporary GeoTIFF (SAMGeo requires file input), reads back binary mask
- Input: tile data + transform + CRS; Output: `dict[str, NDArray[float32]]`

### 3.4 Mask-to-Vector Conversion

Probability masks are converted to georeferenced polygons through:

1. **Thresholding:** Binary mask at probability >= 0.5
2. **Vectorization:** `rasterio.features.shapes` produces GeoJSON-like geometry dicts
3. **Validation:** Filter out invalid or empty polygons via Shapely
4. **Confidence scoring:** Mean probability within the detection bounding region, scaled to 0--100
5. **Valid mask application:** `prob_mask * tile.valid_mask` zeroes out padded regions before vectorization

### 3.5 Post-Processing Pipeline

| Step                    | Method                                                          |
|-------------------------|-----------------------------------------------------------------|
| Non-Maximum Suppression | STRtree spatial index; per-class IoU suppression, higher confidence wins |
| Confidence filter       | Remove detections below threshold (default 50)                   |
| Size filter             | Min/max area in m2 via `Geod(ellps="WGS84")`                    |
| Shape filter            | Remove slivers with aspect ratio > 100                           |
| Simplify                | Douglas-Peucker with tolerance converted from meters to degrees  |
| AOI clip                | Clip geometries to AOI boundary                                  |

Each step reports counts for pipeline transparency.

### 3.6 Workflow Orchestration

**Temporal** provides production-grade workflow management:

- Each pipeline step is a Temporal **activity** with automatic retry and timeout
- **Checkpoint data** persists per activity in the `ProcessingJob` database row for idempotency
- Activities can be individually retried without re-running the entire pipeline
- The CLI offers a `--use-temporal` flag for production mode, or runs locally for development

---

## 4. Object Classes

| Class      | Model    | Geometry | Justification                                                    |
|------------|----------|----------|------------------------------------------------------------------|
| Building   | SAMGeo   | Polygon  | Distinct spatial footprints; SAM excels at instance segmentation |
| Road       | TorchGeo | Polygon  | Linear features well-captured by semantic segmentation           |
| Vegetation | TorchGeo | Polygon  | Strong spectral signature in RGB, large contiguous areas         |
| Water      | TorchGeo | Polygon  | Distinct spectral response, typically uniform regions            |

Class selection is justified by:
- **Data quality:** RGB imagery at sub-meter to meter resolution supports all four classes
- **Model capability:** TorchGeo handles multi-class semantic segmentation; SAMGeo provides superior instance boundaries for buildings
- **Practical utility:** These classes cover the primary needs for territory monitoring and infrastructure inventory

---

## 5. Quality Metrics

### 5.1 Evaluation Methodology

The `QualityEvaluatorService` evaluates detection quality using a greedy IoU-based matching algorithm:

1. Compute IoU between all prediction-ground truth pairs within the same class
2. Greedily match pairs by descending IoU (minimum IoU threshold: 0.5)
3. Unmatched predictions are **false positives**; unmatched ground truth are **false negatives**

### 5.2 Per-Class Metrics

| Metric    | Formula                                                     |
|-----------|-------------------------------------------------------------|
| Precision | TP / (TP + FP)                                              |
| Recall    | TP / (TP + FN)                                              |
| F1 Score  | 2 * Precision * Recall / (Precision + Recall)               |
| IoU       | Mean IoU of matched pairs                                   |
| mAP       | Average Precision via 11-point interpolation                |

### 5.3 Error Analysis

The system collects error examples with geometry WKT for debugging:

- **False positive examples:** Detections with no matching ground truth, including geometry and confidence
- **False negative examples:** Ground truth objects with no matching prediction, including geometry

### 5.4 Common Error Sources

| Error Source         | Impact                                              |
|----------------------|-----------------------------------------------------|
| Resolution           | Sub-meter required for small buildings; road edges lost at lower resolution |
| Shadows              | Cast shadows may be misclassified as water or cause missed buildings       |
| Seasonal effects     | Vegetation extent varies; deciduous trees may appear as bare ground        |
| Mixed land cover     | Transition zones between classes produce uncertain predictions             |

### 5.5 Control Sample

The `create_control_sample()` method selects a random subset of detections for manual review, enabling quality estimation without full labeling.

---

## 6. Zone Indicators

The `IndicatorCalculatorService` computes per-zone quantitative metrics:

| Indicator         | Computation                                                   |
|-------------------|---------------------------------------------------------------|
| Count             | Number of detections per class intersecting the zone          |
| Density (per km2) | Count / geodesic zone area in km2                             |
| Total area (m2)   | Sum of geodesic detection areas, clipped to zone boundary     |

- Partial intersections are handled by clipping detection geometry to the zone boundary
- Zone area uses `Geod(ellps="WGS84").geometry_area_perimeter()` for geodesic accuracy
- Results are exported as CSV and JSON with optional summary table grouped by zone

---

## 7. GIS Output

### 7.1 Supported Formats

| Format      | File Extension | Notes                                       |
|-------------|----------------|---------------------------------------------|
| GeoJSON     | `.geojson`     | Mandatory; always WGS84 per RFC 7946        |
| GeoPackage  | `.gpkg`        | Single-file, supports complex schemas       |
| Shapefile   | `.shp`         | Legacy compatibility, widely supported      |

### 7.2 GIS Compatibility

- All exports are validated to open correctly in QGIS
- CRS is embedded in each export format
- Geometries align with base maps without manual adjustment
- `area_m2` is computed via geodesic methods for accuracy at any latitude

---

## 8. Conclusions

1. **Complete pipeline:** Terrascope implements the full workflow from raw GeoTIFF to GIS-ready output with quantitative indicators
2. **Modular architecture:** Each service (imagery, tiling, detection, post-processing, export, indicators, evaluation) is independently testable and replaceable
3. **Production readiness:** Temporal workflows provide resumability, retry logic, and checkpoint-based idempotency
4. **Quality assurance:** Built-in evaluation framework with standard metrics, error analysis, and control sampling
5. **GIS standards:** CRS propagation, geodesic accuracy, and RFC 7946-compliant GeoJSON ensure interoperability

---

## 9. Limitations

1. **Pretrained models without fine-tuning:** Models are used out-of-the-box without domain-specific training; fine-tuning on target imagery would improve accuracy
2. **RGB-only input:** The system processes 3-band RGB imagery; multispectral data (NIR, SWIR) would significantly improve vegetation and water classification
3. **No multi-temporal change detection:** Each processing run is independent; the `change_flag` output field is reserved but not implemented
4. **Resolution dependency:** Sub-meter imagery is required for reliable building detection; lower resolution degrades all classes
5. **SAMGeo file I/O:** SAMGeo requires writing temporary GeoTIFF files, adding latency compared to in-memory processing
6. **Uncalibrated confidence scores:** Confidence is the mean probability within the bounding region, not a calibrated posterior probability
7. **Geographic coordinate IoU:** NMS uses IoU computed on geographic (lat/lon) coordinates, which introduces minor distortion compared to projected coordinates at large scales
8. **No GPU optimization:** Default configuration uses CPU inference; GPU support requires CUDA-enabled PyTorch installation
9. **Single-date processing:** No temporal aggregation or consensus across multiple acquisition dates
