import typer

from cli.commands import evaluate, process, stac, worker

app = typer.Typer(name="terrascope", help="Terrascope: satellite imagery analysis CLI")

app.add_typer(process.app, name="process")
app.add_typer(stac.app, name="stac")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(worker.app, name="worker")

if __name__ == "__main__":
    app()
