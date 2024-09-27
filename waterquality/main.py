import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import typer
from typing import Optional
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
    model_name: Annotated[str, typer.Argument()] = "model.keras"
    ):
    """
    Load data files, clean and merge data sources, and finally create a model
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
    pipeline.save_model(pipeline.model.model, f"{base_models_path}/{model_name}")
    print_log(f'Model saved at "{base_models_path}/{model_name}"')


@app.command()
def load_evaluate_model(
    sensor_data_file: Annotated[str, typer.Argument()], 
    weather_data_file: Annotated[str, typer.Argument()], 
    model_path: Annotated[str, typer.Argument()],
    save_plot_path: Annotated[Optional[str], typer.Argument()] = None,
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
    pipeline.evaluate_model(plot=True, save_plot_path=save_plot_path)
    if save_plot_path is not None: print_log(f'Saved plot at "{save_plot_path}"')

@app.command()
def hyperparameter_tuning(
    sensor_data_file: Annotated[str, typer.Argument()],
    weather_data_file: Annotated[str, typer.Argument()],
    timesteps_in: Annotated[int, typer.Argument()],
    timesteps_out: Annotated[int, typer.Argument()],
    training_epochs: Annotated[int, typer.Argument()] = 10,
    model_name: Annotated[str, typer.Argument()] = "tuned-model.keras",
    batch_size: Annotated[int, typer.Argument()] = 72,
    max_trials: Annotated[int, typer.Argument()] = 5,
    executions_per_trial: Annotated[int, typer.Argument()] = 3,
    ):
    """
    Hyperparameter tuning
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
    print_log(f"Hyperparameter tuning started...")
    pipeline.hyperparameter_tuning(
        epochs=training_epochs, 
        batch_size=batch_size, 
        max_trials=max_trials, 
        executions_per_trial=executions_per_trial
    )
    print_log(f"Hyperparameter tuning done...")
    pipeline.save_model(pipeline.best_model, f"{base_models_path}/{model_name}")
    print_log(f'Model saved at "{base_models_path}/{model_name}"')

@app.command()
def predict_random_sample(
    sensor_data_file: Annotated[str, typer.Argument()],
    weather_data_file: Annotated[str, typer.Argument()],
    model_path: Annotated[str, typer.Argument()],
    save_plot_path: Annotated[Optional[str], typer.Argument()] = None,
    ):
    """
    Load data files, clean and merge data sources, load model and finally make a prediction
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
    timesteps_in = pipeline.model.model.input_shape[1]
    timesteps_out = pipeline.model.model.output_shape[1]
    pipeline.create_dataset(timesteps_in=timesteps_in, timesteps_out=timesteps_out)
    print_log(f"Preprocessing done...")
    print_log(f"Predicting next {timesteps_out} timesteps using as input previous {timesteps_in} timesteps")
    x, y = pipeline.pick_random_sample()
    if timesteps_out == 1:
        r = pipeline.predict_single_value(x, y, plot=True, save_plot_path=save_plot_path)
    else:
        r = pipeline.predict_sequence_values(x, y, plot=True, save_plot_path=save_plot_path)
    print_log(f"Prediction/s: {r}")
    if save_plot_path is not None: print_log(f"Prediction/s plot saved at: {save_plot_path}")