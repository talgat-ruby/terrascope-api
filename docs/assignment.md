# Technical Assignment

## Case Study: Extraction of Objects and Features from Satellite Imagery with GIS Layer Generation

---

## 1. Purpose

Participants are asked to develop a prototype software solution that, based on Earth remote sensing data (satellite imagery), automatically detects objects and spatial features on the terrain and generates results in geoinformation formats for further use in GIS systems and geoportals.

The solution is considered as a component that can be applied to tasks such as:

- territory monitoring
- inventory and condition control of objects
- obtaining quantitative indicators for specified zones

---

## 2. Objective

Create a prototype that provides:

1. processing of satellite imagery within a defined territory
2. automatic extraction of objects with coordinate reference
3. export of results as geospatial layers and aggregated indicators

---

## 3. Input Data

Organizers provide the following dataset:

- satellite images in **GeoTIFF** format (or similar) with georeferencing
- boundaries of the area of interest (AOI) in polygon format
- additional data (if available):
  - DEM / elevation
  - administrative boundaries
  - base map examples

If part of the source materials does not contain georeferencing, the team must:

- describe the chosen method for alignment to the coordinate system
- specify accuracy limitations

---

## 4. Result Requirements

### 4.1 Object and Feature Extraction

The solution must automatically detect spatial entities in images.

Allowed geometry types:

- point objects
- linear objects
- polygonal contours / zones

The team independently determines the set of classes (categories) to be recognized within the prototype.

Selected classes must be:

- described
- justified based on data quality
- validated with a defined evaluation approach

---

### 4.2 Indicators by Territory

The solution must generate quantitative indicators for specified zones, for example:

- number of detected objects per class
- density of objects per unit area
- total area of zones according to selected attributes
- change assessment when data from multiple dates is available

The methodology for calculating indicators must be:

- described in the report
- supported with examples

---

## 5. GIS Output Format (Mandatory)

Recognition results must be exported in GIS formats:

- **GeoJSON** (mandatory)
- recommended:
  - GeoPackage
  - Shapefile

Each record in the resulting layer must contain:

| Field | Description |
|------|-------------|
| id | unique identifier |
| class | class or category |
| confidence | confidence score (0–100) or other agreed metric |
| source | identifier of the original scene |
| geometry | Point / Polygon |
| optional fields | area_m2, length_m, date, change_flag |

Coordinate requirements:

- layer must open correctly in GIS applications (e.g. QGIS)
- must align with base map without manual adjustment
- coordinate reference system must be specified (EPSG)

---

## 6. Data Processing Requirements

1. territory processing must support tiling with subsequent merging of results
2. false positive reduction measures must be implemented:
   - filtering by size
   - filtering by shape
   - filtering by context
   - duplicate removal
   - post-processing rules
3. instructions for running the solution must be provided:
   - dependencies
   - versions
   - command sequence

Containerization (**Docker**) is recommended but not required.

---

## 7. Quality Evaluation

The team must provide an evaluation of result quality.

Two scenarios are allowed:

### With labeled data

Calculate standard metrics:

- Precision
- Recall
- F1 score
- mAP
- IoU

### Without labeled data

- create a control sample
- manually verify several areas
- calculate metrics on this subset

The report must include:

- selected metrics and justification
- final metric values
- examples of typical errors (2–3 cases)
- explanation of causes:
  - resolution issues
  - seasonal effects
  - shadows
  - mixed land cover

---

## 8. Expected Solution Structure

Recommended workflow:

1. image loading and preparation
2. tile-based processing
3. prediction generation
4. post-processing:
   - merging
   - filtering
   - duplicate removal
5. export to GeoJSON / GeoPackage
6. visual verification:
   - map
   - screenshots
   - QGIS project

---

## 9. Deliverables

The team must provide:

### 1. Source code

- repository
- run instructions

### 2. Technical report

Includes:

- data description
- approach
- metrics
- conclusions

### 3. Demonstration results

- at least 2 example territories with exported GeoJSON layers
- summary table of indicators per zone
- visualization of layer overlay on base map:
  - screenshots
  - QGIS project

### 4. Presentation

Recommended structure:

- objective
- data
- approach
- quality evaluation
- results
- limitations

---

## 10. Evaluation Criteria

1. correctness of geospatial output:
   - CRS correctness
   - geometry validity
   - attributes completeness
   - GIS compatibility

2. recognition quality:
   - metrics clarity
   - validation examples

3. engineering quality:
   - tiling implementation
   - post-processing approach
   - reproducibility

4. report clarity

5. understanding of limitations and applicability conditions
