import asyncio
from pathlib import Path

import typer

from core.services.stac import StacService

app = typer.Typer(help="Search and download satellite imagery via STAC")


@app.command("search")
def search(
    bbox: str = typer.Option(..., help="Bounding box: minx,miny,maxx,maxy"),
    datetime_range: str = typer.Option(
        None, "--datetime", help="Date range: start/end"
    ),
    collection: str = typer.Option("sentinel-2-l2a", help="STAC collection"),
) -> None:
    """Search for available satellite imagery."""
    bbox_floats = tuple(float(x) for x in bbox.split(","))
    if len(bbox_floats) != 4:
        typer.echo("Error: bbox must have exactly 4 values (minx,miny,maxx,maxy)")
        raise typer.Exit(code=1)

    stac = StacService()
    items = asyncio.run(
        stac.search(bbox_floats, datetime_range or "", collection)  # type: ignore[arg-type]
    )

    typer.echo(f"Found {len(items)} items:")
    for item in items:
        dt = item.datetime.isoformat() if item.datetime else "N/A"
        assets = ", ".join(item.assets.keys())
        typer.echo(f"  {item.id}  {dt}  assets=[{assets}]")


@app.command("download")
def download(
    item_id: str = typer.Option(..., help="STAC item ID"),
    bbox: str = typer.Option(
        ..., help="Bounding box used for search: minx,miny,maxx,maxy"
    ),
    datetime_range: str = typer.Option(
        None, "--datetime", help="Date range: start/end"
    ),
    collection: str = typer.Option("sentinel-2-l2a", help="STAC collection"),
    asset_key: str = typer.Option("visual", "--asset", help="Asset key to download"),
    output: Path = typer.Option("./data", "--output", "-o", help="Output directory"),
) -> None:
    """Download a STAC scene."""
    bbox_floats = tuple(float(x) for x in bbox.split(","))
    if len(bbox_floats) != 4:
        typer.echo("Error: bbox must have exactly 4 values")
        raise typer.Exit(code=1)

    async def _download() -> Path:
        stac = StacService()
        items = await stac.search(bbox_floats, datetime_range or "", collection)  # type: ignore[arg-type]
        item = next((i for i in items if i.id == item_id), None)
        if item is None:
            typer.echo(f"Error: item '{item_id}' not found in search results")
            raise typer.Exit(code=1)
        return await stac.download(item, output, asset_key=asset_key)

    path = asyncio.run(_download())
    typer.echo(f"Downloaded to: {path}")
