import typer

from cli.commands import db, process, stac

app = typer.Typer(name="terrascope", help="Terrascope: satellite imagery analysis CLI")

app.add_typer(process.app, name="process")
app.add_typer(stac.app, name="stac")
app.add_typer(db.app, name="db")

if __name__ == "__main__":
    app()
