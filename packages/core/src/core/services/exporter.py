"""GISExporterService -- GeoJSON export for bbox detections.

Geometry is the bbox polygon (so QGIS / geojson.io draw rectangles by
default). Centroid is included as a WKT attribute for callers that want a
single point per detection. PNG overlay is the headline visual deliverable
(`core.detection.render_overlay`).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from core.detection.types import Detection


class GISExporterService:
    _COLUMNS = ["id", "class_name", "confidence", "geometry", "centroid_wkt"]

    def to_geodataframe(
        self, detections: list[Detection], crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        if not detections:
            return gpd.GeoDataFrame(columns=self._COLUMNS, geometry="geometry", crs=crs)

        records = [
            {
                "id": d.id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "geometry": box(*d.bbox),
                "centroid_wkt": d.centroid.wkt,
            }
            for d in detections
        ]
        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    def export_geojson(
        self,
        detections: list[Detection],
        path: str | Path,
        crs: str = "EPSG:4326",
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf = self.to_geodataframe(detections, crs)
        out = gdf.to_crs("EPSG:4326") if str(gdf.crs) != "EPSG:4326" else gdf
        out.to_file(path, driver="GeoJSON")
        return path
