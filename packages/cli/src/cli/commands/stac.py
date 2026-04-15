from pathlib import Path

import typer

app = typer.Typer(help="Search and download satellite imagery via STAC")


@app.command("search")
def search(
    bbox: str = typer.Option(..., help="Bounding box: minx,miny,maxx,maxy"),
    datetime_range: str = typer.Option(None, "--datetime", help="Date range: start/end"),
    collection: str = typer.Option("sentinel-2-l2a", help="STAC collection"),
) -> None:
    """Search for available satellite imagery."""
    # TODO: Implement in Phase 2
    typer.echo(f"Searching STAC: bbox={bbox}, datetime={datetime_range}, collection={collection}")


@app.command("download")
def download(
    item_id: str = typer.Option(..., help="STAC item ID"),
    output: Path = typer.Option("./data", "--output", "-o", help="Output directory"),
) -> None:
    """Download a STAC scene."""
    # TODO: Implement in Phase 2
    typer.echo(f"Downloading {item_id} to {output}")
