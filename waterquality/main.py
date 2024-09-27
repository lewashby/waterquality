import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import typer
from typing_extensions import Annotated
from .Pipeline import Pipeline

app = typer.Typer(no_args_is_help=True)
base_models_path = "waterquality/models"

def print_log(msg: str):
    text = typer.style(msg, fg=typer.colors.RED, bold=True)
    typer.echo(text)

@app.callback()
def callback():
    """
    Water Quality Management
    """


@app.command()
def pipeline(
    sensor_data_file: Annotated[str, typer.Argument()], 
    weather_data_file: Annotated[str, typer.Argument()], 
    timesteps_in: Annotated[int, typer.Argument()], 
    timesteps_out: Annotated[int, typer.Argument()],
    training_epochs: Annotated[int, typer.Argument()] = 10,
    save_model_path: Annotated[str, typer.Argument()] = "model.keras"
    ):
    """
    Load data files, clean and merge data soruces, and finally create a model
    """
    pipeline = Pipeline(sensor_data_file, weather_data_file)
    pipeline.load_files()
    print_log(f"Loaded files...")
    pipeline.slice_data()
    pipeline.add_doc_formula_values()
    pipeline.handle_missing_values()
    pipeline.drop_columns()
    pipeline.merge_data_sources()
    pipeline.align_start_date()
    print_log(f"Preprocessing done...")
    pipeline.create_dataset(timesteps_in=timesteps_in, timesteps_out=timesteps_out)
    print_log(f"Dataset created...")
    pipeline.create_model(output=timesteps_out)
    print_log(f"Model created...")
    print_log(f"Training model...")
    pipeline.train_model(epochs=training_epochs)
    print_log(f"Model trained...")
    print_log(f"Evaluating model...")
    pipeline.evaluate_model()
    pipeline.save_model(f"{base_models_path}/{save_model_path}")
    print_log(f'Model saved at "{base_models_path}/{save_model_path}"')


@app.command()
def load_evaluate_model(
    sensor_data_file: Annotated[str, typer.Argument()], 
    weather_data_file: Annotated[str, typer.Argument()], 
    model_path: Annotated[str, typer.Argument()],
    ):
    """
    Load pretrained model and evaluate data with it
    """
    pipeline = Pipeline(sensor_data_file, weather_data_file)
    pipeline.load_trained_model(model_path)
    print_log(f"Model loaded...")
    pipeline.load_files()
    print_log(f"Loaded files...")
    pipeline.slice_data()
    pipeline.add_doc_formula_values()
    pipeline.handle_missing_values()
    pipeline.drop_columns()
    pipeline.merge_data_sources()
    pipeline.align_start_date()
    print_log(f"Preprocessing done...")
    timesteps_in = pipeline.model.input_shape[1]
    timesteps_out = pipeline.model.output_shape[1]
    pipeline.create_dataset(timesteps_in=timesteps_in, timesteps_out=timesteps_out)
    print_log(f"Dataset created...")
    print_log(f"Evaluating model...")
    pipeline.evaluate_model()