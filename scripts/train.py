import click
import numpy as np
import pandas as pd
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common.gen_features import *
from common.generators import train_feature_set
from common.model_store import *
from service.App import *


class P:
    """Defines parameters for controlling data processing and storage options."""
    in_nrows = 100_000_000  # Number of rows to load (for debugging)
    tail_rows = 0  # Number of last rows to select (for debugging)
    store_predictions = True  # Flag to indicate whether to store predictions


@click.command()
@click.option('--config_file', '-c', type=click.Path(exists=True), default='configs/config.json', help='Path to configuration file')
def main(config_file):
    """Main function for loading data, training models, and storing outputs as specified in the configuration."""
    load_config(config_file)
    time_column = App.config["time_column"]
    now = datetime.now()

    data_path = Path(App.config["data_folder"])

    algorithms = App.config.get("algorithms")
    training_directory = App.config.get("training_directory")

    # Change working directory to training_directory
    if training_directory:
        try:
            os.chdir(training_directory)
            print(f"Changed working directory to: {training_directory}")
        except Exception as e:
            print(f"ERROR: Failed to change directory to '{training_directory}'. Exception: {e}")
            sys.exit(1)
    else:
        print("WARNING: 'training_directory' not specified in configuration. Skipping directory change.")

    # Execute each algorithm script
    if algorithms:
        for algo_script in algorithms:
            algo_path = r"./scripts" / Path(algo_script)
            if not algo_path.is_absolute():
                algo_path = Path(training_directory) / algo_script

            if not algo_path.is_file():
                print(f"ERROR: Algorithm script '{algo_path}' does not exist.")
                continue

            if not os.access(algo_path, os.X_OK):
                print(f"ERROR: Algorithm script '{algo_path}' is not executable.")
                continue

            print(f"Executing algorithm script: {algo_path}")
            try:
                result = subprocess.run([str(algo_path)], check=True, capture_output=True, text=True)
                print(f"Output of '{algo_script}':\n{result.stdout}")
                if result.stderr:
                    print(f"Error Output of '{algo_script}':\n{result.stderr}")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Script '{algo_script}' failed with return code {e.returncode}.")
                print(f"Output:\n{e.output}")
                print(f"Error Output:\n{e.stderr}")
            except Exception as e:
                print(f"ERROR: An unexpected error occurred while executing '{algo_script}': {e}")
    else:
        print("No algorithms specified to execute.")

    print(f"Model training and algorithm execution completed in {str(datetime.now() - now).split('.')[0]}")


if __name__ == '__main__':
    main()
