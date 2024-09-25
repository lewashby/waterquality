import typer


app = typer.Typer(no_args_is_help=True)


@app.callback()
def callback():
    """
    Water Quality Management
    """


@app.command()
def load(sensor_data_file: str, weather_data_file: str):
    """
    Load data files
    """
    typer.echo(f"loaded files")
