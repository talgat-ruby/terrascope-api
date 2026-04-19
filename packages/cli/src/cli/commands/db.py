import importlib.resources

import typer
from alembic import command
from alembic.config import Config

import core

app = typer.Typer(help="Database operations")


@app.command()
def upgrade(revision: str = typer.Argument("head", help="Revision to upgrade to")):
    """Upgrade database to a specific revision (default: head)."""
    ini_path = importlib.resources.files(core) / "alembic.ini"
    script_location = importlib.resources.files(core) / "alembic"

    alembic_cfg = Config(str(ini_path))
    alembic_cfg.set_main_option("script_location", str(script_location))

    typer.echo(f"Running migrations to {revision}...")
    command.upgrade(alembic_cfg, revision)
    typer.echo("Database upgrade complete.")
