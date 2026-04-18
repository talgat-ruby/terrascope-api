"""Imagery loading and tiling activities."""

import asyncio
import json

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import shape as shapely_shape
from temporalio import activity

from core.config import settings
from core.database import async_session_factory
from core.models.processing import JobStatus
from core.services.imagery import ImageryLoaderService
from core.services.tiler import TilerService
from worker.activities._helpers import fail_job, get_job, update_job


@activity.defn
async def load_imagery(job_id: str) -> dict:
    """Load and validate GeoTIFF imagery, save metadata to DB."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)
        await update_job(
            session, job, status=JobStatus.LOADING, current_step="load_imagery"
        )

        try:
            loader = ImageryLoaderService()

            dataset = await asyncio.to_thread(loader.load, job.input_path)
            try:
                metadata = await asyncio.to_thread(loader.get_metadata, dataset)

                aoi_geojson = job.config["aoi"]  # type: ignore[index]
                aoi_crs = job.config.get("aoi_crs", "EPSG:4326")  # type: ignore[union-attr]
                aoi_geom = shapely_shape(aoi_geojson)

                data, transform, crs = await asyncio.to_thread(
                    loader.clip_to_aoi, dataset, aoi_geom, aoi_crs
                )
            finally:
                dataset.close()

            # Save clipped data to disk
            job_dir = settings.output_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            clipped_path = job_dir / "clipped.npy"
            await asyncio.to_thread(np.save, str(clipped_path), data)

            transform_list = [
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ]

            await update_job(
                session,
                job,
                status=JobStatus.LOADING,
                current_step="load_imagery",
                checkpoint_update={
                    "load": {
                        "metadata": metadata,
                        "crs": crs,
                        "clipped_path": str(clipped_path),
                        "transform": transform_list,
                        "shape": list(data.shape),
                    }
                },
            )

            activity.logger.info(
                f"Loaded imagery for job {job_id}: {metadata['band_count']} bands, "
                f"CRS={crs}, shape={data.shape}"
            )
            return {"status": "loaded", "job_id": job_id, "crs": crs}

        except Exception as e:
            await fail_job(session, job, str(e))
            raise


@activity.defn
async def tile_imagery(job_id: str) -> dict:
    """Generate tiles from imagery, store tile manifest as checkpoint."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)
        await update_job(
            session, job, status=JobStatus.TILING, current_step="tile_imagery"
        )

        try:
            load_checkpoint = job.checkpoint_data["load"]  # type: ignore[index]
            clipped_path = load_checkpoint["clipped_path"]
            transform_list = load_checkpoint["transform"]
            crs = load_checkpoint["crs"]

            data = await asyncio.to_thread(np.load, clipped_path)
            transform = Affine(*transform_list)

            tile_size = (job.config or {}).get("tile_size", settings.tile_size)
            tile_overlap = (job.config or {}).get("tile_overlap", settings.tile_overlap)

            tiler = TilerService(tile_size=tile_size, overlap=tile_overlap)
            tiles = await asyncio.to_thread(
                lambda: list(tiler.generate_tiles(data, transform, crs=crs))
            )

            # Save tiles to disk
            tiles_dir = settings.output_dir / job_id / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)

            manifest = []
            for tile in tiles:
                tile_name = f"tile_{tile.index[0]}_{tile.index[1]}"
                np.save(str(tiles_dir / f"{tile_name}_data.npy"), tile.data)
                np.save(str(tiles_dir / f"{tile_name}_mask.npy"), tile.valid_mask)

                tile_transform = tile.transform
                manifest.append(
                    {
                        "index": list(tile.index),
                        "pixel_window": list(tile.pixel_window),
                        "transform": [
                            tile_transform.a,
                            tile_transform.b,
                            tile_transform.c,
                            tile_transform.d,
                            tile_transform.e,
                            tile_transform.f,
                        ],
                        "crs": tile.crs,
                        "name": tile_name,
                    }
                )

            manifest_path = tiles_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            await update_job(
                session,
                job,
                status=JobStatus.TILING,
                current_step="tile_imagery",
                checkpoint_update={
                    "tile": {
                        "tile_count": len(tiles),
                        "manifest_path": str(manifest_path),
                        "tiles_dir": str(tiles_dir),
                    }
                },
            )

            activity.logger.info(f"Tiled imagery for job {job_id}: {len(tiles)} tiles")
            return {"status": "tiled", "job_id": job_id, "tile_count": len(tiles)}

        except Exception as e:
            await fail_job(session, job, str(e))
            raise
