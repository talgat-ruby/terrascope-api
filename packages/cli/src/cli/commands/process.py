from pathlib import Path

import typer

app = typer.Typer(help="Process satellite imagery")


@app.callback(invoke_without_command=True)
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Path to GeoTIFF file"),
    aoi: Path = typer.Option(..., "--aoi", "-a", help="Path to AOI GeoJSON file"),
    output: Path = typer.Option("./output", "--output", "-o", help="Output directory"),
    use_temporal: bool = typer.Option(False, "--use-temporal", help="Submit to Temporal instead of running locally"),
) -> None:
    """Process satellite imagery and extract features."""
    # TODO: Implement in Phase 8
    typer.echo(f"Processing {input} with AOI {aoi}")
    typer.echo(f"Output directory: {output}")
    typer.echo(f"Using Temporal: {use_temporal}")
    typer.echo("Not yet implemented -- coming in Phase 8")
