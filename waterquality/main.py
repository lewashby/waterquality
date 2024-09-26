import typer
from .Pipeline import Pipeline

app = typer.Typer(no_args_is_help=True)

@app.callback()
def callback():
    """
    Water Quality Management
    """


@app.command()
def pipeline(sensor_data_file: str, weather_data_file: str, timesteps_in: int, timesteps_out: int):
    """
    Load data files, clean and merge data soruces, and finally create a model
    """
    pipeline = Pipeline(sensor_data_file, weather_data_file)
    pipeline.load_files()
    typer.echo(f"loaded files")