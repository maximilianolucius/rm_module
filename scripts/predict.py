import click
import numpy as np
import pandas as pd

from common.generators import predict_feature_set
from common.model_store import *
from service.App import *

"""
Apply models to (previously generated) features and compute prediction scores.
"""

class P:
    in_nrows = 100_000_000  # Maximum number of rows to load (for debugging)
    tail_rows = 0  # Number of last rows to select (for debugging)

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)
    time_column = App.config["time_column"]
    now = datetime.now()

    # Load feature matrix
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    file_path = data_path / App.config.get("matrix_file_name")

    if not file_path.is_file():
        print(f"ERROR: Input file does not exist: {file_path}")
        return

    print(f"Loading data from {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unsupported file extension '{file_path.suffix}'. Only 'csv' and 'parquet' are supported.")
        return
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:].reset_index(drop=True)
    print(f"Data size after filtering: {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    # Prepare data by selecting columns and rows
    train_features = App.config.get("train_features")
    labels = App.config["labels"]
    algorithms = App.config.get("algorithms")
    out_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [col for col in out_columns if col in df.columns]
    labels_present = set(labels).issubset(df.columns)
    all_features = train_features + (labels if labels_present else [])
    df = df[out_columns + [col for col in all_features if col not in out_columns]]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=train_features).reset_index(drop=True)

    if len(df) == 0:
        print("ERROR: Empty dataset after removing NULLs in feature columns.")
        return

    # Load models
    model_path = Path(App.config["model_folder"]).resolve()
    models = load_models(model_path, labels, algorithms)

    # Generate/predict train features
    train_feature_sets = App.config.get("train_feature_sets", [])
    if not train_feature_sets:
        print("ERROR: No train feature sets defined.")
        return

    print(f"Start generating trained features for {len(df)} records.")
    out_df = pd.DataFrame()
    features = []
    scores = {}

    for i, fs in enumerate(train_feature_sets):
        fs_now = datetime.now()
        print(f"Processing train feature set {i + 1}/{len(train_feature_sets)}: {fs.get('generator')}...")

        fs_out_df, fs_features, fs_scores = predict_feature_set(df, fs, App.config, models)
        out_df = pd.concat([out_df, fs_out_df], axis=1)
        features.extend(fs_features)
        scores.update(fs_scores)

        fs_elapsed = datetime.now() - fs_now
        print(f"Finished set {i + 1}/{len(train_feature_sets)}. Time: {str(fs_elapsed).split('.')[0]}")

    # Store scores
    if labels_present:
        metrics_file_name = "prediction-metrics.txt"
        metrics_path = data_path / metrics_file_name
        with open(metrics_path, 'a+') as f:
            f.write("\n".join([f"{col}, {score}" for col, score in scores.items()]) + "\n\n")
        print(f"Metrics stored in: {metrics_path.absolute()}")

    # Store predictions
    out_df = out_df.join(df[out_columns + (labels if labels_present else [])])
    out_path = data_path / App.config.get("predict_file_name")

    print(f"Storing predictions with {len(out_df)} records and {len(out_df.columns)} columns to {out_path}...")
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format='%.6f')
    else:
        print(f"ERROR: Unsupported file extension '{out_path.suffix}'. Only 'csv' and 'parquet' are supported.")
        return

    print(f"Predictions stored in: {out_path}")

    elapsed = datetime.now() - now
    print(f"Finished in {str(elapsed).split('.')[0]}")

if __name__ == '__main__':
    main()
