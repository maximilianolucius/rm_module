from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor
import click
import numpy as np
import pandas as pd
from service.App import *
from common.utils import *
from common.gen_features import *
from common.classifiers import *
from common.model_store import *


class P:
    """Defines parameters for data processing and prediction configurations."""
    in_nrows = 100_000_000  # Maximum number of records to load


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """Main function for generating rolling label predictions using iterative model training."""
    load_config(config_file)
    time_column = App.config["time_column"]
    now = datetime.now()

    rp_config = App.config["rolling_predict"]
    use_multiprocessing = rp_config.get("use_multiprocessing", False)
    max_workers = rp_config.get("max_workers", None)

    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    file_path = data_path / App.config.get("matrix_file_name")

    # Load data
    print(f"Loading data from {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unsupported file type '{file_path.suffix}'")
        return
    print(f"Loaded {len(df)} records.")

    # Set data range
    data_start = rp_config.get("data_start", 0)
    data_start = find_index(df, data_start) if isinstance(data_start, str) else data_start
    data_end = rp_config.get("data_end", None)
    data_end = find_index(df, data_end) if isinstance(data_end, str) else data_end
    df = df.iloc[data_start:data_end].reset_index(drop=True)

    # Set rolling prediction parameters
    prediction_start, prediction_size, prediction_steps = setup_prediction_params(rp_config, len(df))
    validate_params(df, prediction_start, prediction_size, prediction_steps)

    # Prepare data columns
    label_horizon = App.config["label_horizon"]
    train_features, labels, algorithms = App.config.get("train_features"), App.config["labels"], App.config.get(
        "algorithms")
    out_columns = [time_column, 'open', 'high', 'low', 'close', 'volume', 'close_time']
    out_columns = [col for col in out_columns if col in df.columns]
    df = df[out_columns + train_features + labels].replace([np.inf, -np.inf], np.nan).dropna(subset=labels).reset_index(
        drop=True)

    labels_hat_df = pd.DataFrame()
    print(f"Starting rolling prediction loop with {prediction_steps} steps...")

    for step in range(prediction_steps):
        predict_start, predict_end = prediction_start + step * prediction_size, prediction_start + (
                    step + 1) * prediction_size
        predict_df = df.iloc[predict_start:predict_end]
        predict_labels_df = pd.DataFrame(index=predict_df.index)

        train_df, train_length = setup_training_data(df, label_horizon, train_features, predict_start,
                                                     App.config.get("train_length"))

        if use_multiprocessing:
            execution_results = execute_parallel_predictions(labels, algorithms, train_df, predict_df, train_features,
                                                             max_workers)
            predict_labels_df = process_parallel_results(execution_results, predict_labels_df)
        else:
            predict_labels_df = execute_sequential_predictions(labels, algorithms, train_df, predict_df, train_features,
                                                               predict_labels_df)

        labels_hat_df = pd.concat([labels_hat_df, predict_labels_df])
        print(f"Completed step {step + 1}/{prediction_steps}")

    # Store predictions
    out_df = labels_hat_df.join(df[out_columns + labels])
    store_predictions(out_df, data_path, App.config.get("predict_file_name"))

    # Compute and store performance scores
    score_lines = compute_performance_scores(out_df, labels_hat_df)
    store_performance_scores(score_lines, data_path, App.config.get("predict_file_name"))

    print(f"Finished rolling prediction in {str(datetime.now() - now).split('.')[0]}")


def setup_prediction_params(rp_config, df_length):
    """Determine and validate prediction start, size, and step values."""
    prediction_start = rp_config.get("prediction_start", None)
    prediction_start = find_index(df, prediction_start) if isinstance(prediction_start, str) else prediction_start
    prediction_size, prediction_steps = rp_config.get("prediction_size"), rp_config.get("prediction_steps")

    if not prediction_start:
        prediction_start = df_length - prediction_size * prediction_steps
    elif not prediction_size:
        prediction_size = (df_length - prediction_start) // prediction_steps
    elif not prediction_steps:
        prediction_steps = (df_length - prediction_start) // prediction_size

    return prediction_start, prediction_size, prediction_steps


def validate_params(df, prediction_start, prediction_size, prediction_steps):
    """Ensure sufficient data is available for the specified prediction parameters."""
    if len(df) - prediction_start < prediction_steps * prediction_size:
        raise ValueError("Not enough data for the specified prediction configuration.")


def setup_training_data(df, label_horizon, train_features, predict_start, train_length):
    """Prepare training data with necessary adjustments for the training length."""
    train_end = predict_start - label_horizon - 1
    train_start = max(0, train_end - train_length) if train_length else 0
    train_df = df.iloc[train_start:train_end].dropna(subset=train_features)
    return train_df, train_length


def execute_parallel_predictions(labels, algorithms, train_df, predict_df, train_features, max_workers):
    """Run model predictions in parallel across specified labels and algorithms."""
    execution_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for label in labels:
            for model_config in algorithms:
                execution_results = submit_parallel_job(executor, execution_results, label, model_config, train_df,
                                                        predict_df, train_features)
    return execution_results


def submit_parallel_job(executor, execution_results, label, model_config, train_df, predict_df, train_features):
    """Submit a prediction task for a given label and model configuration."""
    algo_name, algo_type = model_config.get("name"), model_config.get("algo")
    algo_train_length, score_column_name = model_config.get("train", {}).get(
        "length"), label + label_algo_separator + algo_name
    train_df_2 = train_df.tail(algo_train_length) if algo_train_length else train_df
    df_X, df_y, df_X_test = train_df_2[train_features], train_df_2[label], predict_df[train_features]

    if algo_type == "gb":
        execution_results[score_column_name] = executor.submit(train_predict_gb, df_X, df_y, df_X_test, model_config)
    elif algo_type == "nn":
        execution_results[score_column_name] = executor.submit(train_predict_nn, df_X, df_y, df_X_test, model_config)
    elif algo_type == "lc":
        execution_results[score_column_name] = executor.submit(train_predict_lc, df_X, df_y, df_X_test, model_config)
    elif algo_type == "svc":
        execution_results[score_column_name] = executor.submit(train_predict_svc, df_X, df_y, df_X_test, model_config)
    return execution_results


def process_parallel_results(execution_results, predict_labels_df):
    """Retrieve results from parallel execution and populate the prediction DataFrame."""
    for score_column_name, future in execution_results.items():
        predict_labels_df[score_column_name] = future.result()
    return predict_labels_df


def execute_sequential_predictions(labels, algorithms, train_df, predict_df, train_features, predict_labels_df):
    """Run predictions sequentially for each label and model configuration."""
    for label in labels:
        for model_config in algorithms:
            predict_labels_df = make_prediction(label, model_config, train_df, predict_df, train_features,
                                                predict_labels_df)
    return predict_labels_df


def make_prediction(label, model_config, train_df, predict_df, train_features, predict_labels_df):
    """Execute prediction for a given label and model configuration."""
    algo_name, algo_type = model_config.get("name"), model_config.get("algo")
    score_column_name = label + label_algo_separator + algo_name
    train_df_2 = train_df.tail(model_config.get("train", {}).get("length")) if model_config.get("train", {}).get(
        "length") else train_df
    df_X, df_y, df_X_test = train_df_2[train_features], train_df_2[label], predict_df[train_features]

    if algo_type == "gb":
        predict_labels_df[score_column_name] = train_predict_gb(df_X, df_y, df_X_test, model_config)
    elif algo_type == "nn":
        predict_labels_df[score_column_name] = train_predict_nn(df_X, df_y, df_X_test, model_config)
    elif algo_type == "lc":
        predict_labels_df[score_column_name] = train_predict_lc(df_X, df_y, df_X_test, model_config)
    elif algo_type == "svc":
        predict_labels_df[score_column_name] = train_predict_svc(df_X, df_y, df_X_test, model_config)
    return predict_labels_df


def store_predictions(out_df, data_path, file_name):
    """Save the predictions DataFrame to a file."""
    out_path = data_path / file_name
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format='%.6f')
    else:
        print(f"ERROR: Unsupported file type '{out_path.suffix}'")
    print(f"Predictions saved to {out_path}")


def compute_performance_scores(out_df, labels_hat_df):
    """Calculate and return performance scores for each prediction column."""
    score_lines = []
    for score_column_name in labels_hat_df.columns:
        label_column, _ = score_to_label_algo_pair(score_column_name)
        df_scores = pd.DataFrame({"y_true": out_df[label_column], "y_predicted": out_df[score_column_name]}).dropna()
        y_true, y_predicted = df_scores["y_true"].astype(int), df_scores["y_predicted"]
        score = compute_scores(y_true, y_predicted)
        score_lines.append(
            f"{score_column_name}, {score.get('auc'):.3f}, {score.get('ap'):.3f}, {score.get('f1'):.3f}, {score.get('precision'):.3f}, {score.get('recall'):.3f}")
    return score_lines


def store_performance_scores(score_lines, data_path, file_name):
    """Store the computed performance scores to a text file."""
    with open(data_path / file_name.with_suffix('.txt'), "a+") as f:
        f.write("\n".join(score_lines) + "\n\n")
    print("Performance scores stored.")


if __name__ == '__main__':
    main()
