from pathlib import Path
from datetime import datetime
import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.generators import generate_feature_set
from service.App import *

"""
Generate new derived columns according to the signal definitions.
The transformations are applied to the results of ML predictions.
"""

# Parameters
class P:
    in_nrows = 100_000_000  # Maximum rows to load
    start_index = 0         # Starting index for slicing data
    end_index = None        # Ending index for slicing data

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Load configuration and generate derived features based on signal definitions.
    """
    load_config(config_file)
    time_column = App.config["time_column"]
    now = datetime.now()

    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    if not data_path.is_dir():
        print(f"Data folder does not exist: {data_path}")
        return
    out_path = Path(App.config["data_folder"]) / symbol
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists

    # Load data with label point-wise predictions
    file_path = data_path / App.config.get("predict_file_name")
    if not file_path.exists():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading predictions from file: {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unsupported file extension '{file_path.suffix}'. Only 'csv' and 'parquet' are supported.")
        return
    print(f"Predictions loaded. Length: {len(df)}. Width: {len(df.columns)}")

    # Limit data based on parameters
    df = df.iloc[P.start_index:P.end_index].reset_index(drop=True)
    print(f"Data size: {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    # Generate signals
    feature_sets = App.config.get("signal_sets", [])
    if not feature_sets:
        print("ERROR: No signal sets defined.")
        return

    print(f"Generating features for {len(df)} records.")
    all_features = []
    for i, fs in enumerate(feature_sets):
        fs_now = datetime.now()
        print(f"Processing feature set {i + 1}/{len(feature_sets)} with generator {fs.get('generator')}...")
        df, new_features = generate_feature_set(df, fs, last_rows=0)
        all_features.extend(new_features)
        fs_elapsed = datetime.now() - fs_now
        print(f"Finished feature set {i + 1}/{len(feature_sets)}. New features: {len(new_features)}. Time: {str(fs_elapsed).split('.')[0]}")

    print("Feature generation completed.")
    print("Null values per feature:")
    print(df[all_features].isnull().sum().sort_values(ascending=False))

    # Define columns for output
    out_columns = ["timestamp", "open", "high", "low", "close"]
    out_columns.extend(App.config.get('labels'))
    out_columns.extend(all_features)
    out_df = df[out_columns]

    # Store data
    out_path = data_path / App.config.get("signal_file_name")
    print(f"Storing signals in file: {out_path} with {len(out_df)} records and {len(out_df.columns)} columns.")
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format='%.6f')
    else:
        print(f"ERROR: Unsupported file extension '{out_path.suffix}'. Only 'csv' and 'parquet' are supported.")
        return

    print(f"Signals stored in: {out_path}")
    elapsed = datetime.now() - now
    print(f"Signal generation completed in {str(elapsed).split('.')[0]}")

if __name__ == '__main__':
    main()
