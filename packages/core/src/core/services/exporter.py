"""GISExporterService -- export detections to GeoJSON, GeoPackage, Shapefile."""

from pathlib import Path

import geopandas as gpd
from pyproj import Geod

from core.services.detector import RawDetection


class GISExporterService:
    """Exports detections to various GIS formats."""

    def to_geodataframe(
        self, detections: list[RawDetection], crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """Convert detections to a GeoDataFrame.

        Args:
            detections: List of RawDetection instances.
            crs: Coordinate reference system for the output.

        Returns:
            GeoDataFrame with class_name, confidence, source, geometry, area_m2.
        """
        if not detections:
            return gpd.GeoDataFrame(
                columns=["id", "class_name", "confidence", "source", "geometry", "area_m2", "center"],
                geometry="geometry",
                crs=crs,
            )

        records = [
            {
                "id": i,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "source": d.source,
                "geometry": d.geometry,
            }
            for i, d in enumerate(detections)
        ]

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        geod = Geod(ellps="WGS84")
        gdf["area_m2"] = gdf.geometry.apply(
            lambda g: abs(geod.geometry_area_perimeter(g)[0])
        )
        gdf["center"] = gdf.geometry.apply(lambda g: g.centroid.wkt)
        return gdf

    def export_geojson(self, gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
        """Export GeoDataFrame as GeoJSON in WGS84 (RFC 7946).

        Args:
            gdf: GeoDataFrame to export.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        out = gdf.to_crs("EPSG:4326") if str(gdf.crs) != "EPSG:4326" else gdf
        out.to_file(path, driver="GeoJSON")
        return path

    def export_geopackage(self, gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
        """Export GeoDataFrame as GeoPackage.

        Args:
            gdf: GeoDataFrame to export.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(path, driver="GPKG")
        return path

    def export_shapefile(self, gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
        """Export GeoDataFrame as ESRI Shapefile.

        Args:
            gdf: GeoDataFrame to export.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(path, driver="ESRI Shapefile")
        return path

    def export_all(
        self,
        detections: list[RawDetection],
        crs: str = "EPSG:4326",
        output_dir: str | Path = "output",
    ) -> dict[str, Path]:
        """Export detections in all supported formats.

        Args:
            detections: List of RawDetection instances.
            crs: CRS of the detection geometries.
            output_dir: Directory to write output files.

        Returns:
            Dict mapping format name to output file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        gdf = self.to_geodataframe(detections, crs)

        return {
            "geojson": self.export_geojson(gdf, output_dir / "detections.geojson"),
            "gpkg": self.export_geopackage(gdf, output_dir / "detections.gpkg"),
            "shp": self.export_shapefile(gdf, output_dir / "detections.shp"),
        }
