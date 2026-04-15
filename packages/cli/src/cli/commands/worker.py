import typer

app = typer.Typer(help="Temporal worker management")


@app.callback(invoke_without_command=True)
def start() -> None:
    """Start the Temporal worker."""
    from worker.main import main

    typer.echo("Starting Temporal worker...")
    main()
