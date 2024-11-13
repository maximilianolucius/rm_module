from typing import Tuple
from pathlib import Path
import click
import numpy as np
import pandas as pd

from service.App import *
from common.generators import generate_feature_set

class P:
    """
    Parameters for data processing limits.
    """
    in_nrows = 50_000_000  # Maximum number of records to load
    tail_rows = int(10.0 * 525_600)  # Limit processing to the last rows for efficiency

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Main function for feature generation from raw data.
    Loads configuration, reads data, generates derived features, and saves the output.
    """
    load_config(config_file)
    time_column = App.config["time_column"]

    now = datetime.now()

    # Load the merged dataset with a regular time series
    data_path = Path(App.config["data_folder"])
    file_path = data_path / App.config.get("merge_file_name")

    if not file_path.is_file():
        print(f"Data file does not exist: {file_path}")
        return

    print(f"Loading data from source data file {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unsupported file extension '{file_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")
    df = df.iloc[-P.tail_rows:].reset_index(drop=True)
    print(f"Input data size: {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    # Generate derived features
    feature_sets = App.config.get("feature_sets", [])
    if not feature_sets:
        print("ERROR: No feature sets defined in configuration.")
        return

    print(f"Generating features for {len(df)} records...")
    all_features = []
    for i, fs in enumerate(feature_sets):
        fs_now = datetime.now()
        print(f"Processing feature set {i + 1}/{len(feature_sets)} using generator {fs.get('generator')}...")
        df, new_features = generate_feature_set(df, fs, last_rows=0)
        all_features.extend(new_features)
        fs_elapsed = datetime.now() - fs_now
        print(f"Feature set {i + 1} completed. {len(new_features)} features generated in {str(fs_elapsed).split('.')[0]}")

    print("Feature generation completed.")

    print("Number of NULL values in generated features:")
    print(df[all_features].isnull().sum().sort_values(ascending=False))

    # Save the feature matrix to an output file
    out_file_name = App.config.get("feature_file_name")
    out_path = (data_path / out_file_name).resolve()

    print(f"Saving {len(df)} records with {len(df.columns)} columns to output file {out_path}...")
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        print(f"ERROR: Unsupported file extension '{out_path.suffix}'. Only 'csv' and 'parquet' are supported")
        return

    print(f"Output file saved: {out_path} with {len(df)} records")

    # Save feature list to a separate text file
    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f'"{f}"' for f in all_features]) + "\n\n")

    print(f"Feature list with {len(all_features)} features saved to {out_path.with_suffix('.txt')}")

    elapsed = datetime.now() - now
    print(f"Feature generation completed in {str(elapsed).split('.')[0]}. Average time per feature: {str(elapsed / len(all_features)).split('.')[0]}")

if __name__ == '__main__':
    main()
