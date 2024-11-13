from typing import Tuple

import pandas as pd

from common.classifiers import *
from common.model_store import *
from common.gen_features import *
from common.gen_labels_highlow import generate_labels_highlow, generate_labels_highlow2
from common.gen_labels_topbot import generate_labels_topbot, generate_labels_topbot2
from common.gen_signals import (
    generate_smoothen_scores, generate_combine_scores,
    generate_threshold_rule, generate_threshold_rule2
)


def generate_feature_set(df: pd.DataFrame, fs: dict, last_rows: int) -> Tuple[pd.DataFrame, list]:
    """
    Apply the specified feature generator to the input DataFrame based on configuration.
    Returns the DataFrame with new features and a list of generated feature names.
    """
    cp = fs.get("column_prefix")
    f_df = df[df.columns.to_list()]
    if cp:
        f_cols = [col for col in df if col.startswith(cp + "_")]
        f_df = df[f_cols].rename(columns=lambda x: x[len(cp)+1:] if x.startswith(cp) else x)

    generator = fs.get("generator")
    gen_config = fs.get('config', {})
    if generator == "itblib":
        features = generate_features_itblib(f_df, gen_config, last_rows=last_rows)
    elif generator == "depth":
        features = generate_features_depth(f_df)
    elif generator == "tsfresh":
        features = generate_features_tsfresh(f_df, gen_config, last_rows=last_rows)
    elif generator == "talib":
        features = generate_features_talib(f_df, gen_config, last_rows=last_rows)
    elif generator == "itbstats":
        features = generate_features_itbstats(f_df, gen_config, last_rows=last_rows)
    elif generator == "highlow":
        print(f"Generating 'highlow' labels...")
        features = generate_labels_highlow(f_df, horizon=gen_config.get("horizon"))
    elif generator == "highlow2":
        print(f"Generating 'highlow2' labels...")
        f_df, features = generate_labels_highlow2(f_df, gen_config)
    elif generator == "topbot":
        top_level_fracs = [0.01, 0.02, 0.03, 0.04, 0.05]
        f_df, features = generate_labels_topbot(f_df, gen_config.get("columns", "close"), top_level_fracs, [-x for x in top_level_fracs])
    elif generator == "topbot2":
        f_df, features = generate_labels_topbot2(f_df, gen_config)
    elif generator == "smoothen":
        f_df, features = generate_smoothen_scores(f_df, gen_config)
    elif generator == "combine":
        f_df, features = generate_combine_scores(f_df, gen_config)
    elif generator == "threshold_rule":
        f_df, features = generate_threshold_rule(f_df, gen_config)
    elif generator == "threshold_rule2":
        f_df, features = generate_threshold_rule2(f_df, gen_config)
    else:
        generator_fn = resolve_generator_name(generator)
        if not generator_fn:
            raise ValueError(f"Unknown feature generator name: {generator}")
        f_df, features = generator_fn(f_df, gen_config)

    f_df = f_df[features]
    fp = fs.get("feature_prefix")
    if fp:
        f_df = f_df.add_prefix(fp + "_")

    df.drop(list(set(df.columns) & set(f_df.columns)), axis=1, inplace=True)
    df = df.join(f_df)

    return df, f_df.columns.to_list()


def predict_feature_set(df, fs, config, models: dict):
    """
    Predict outcomes for a feature set using pre-trained models and specified algorithms.
    Returns predictions, the feature names, and a score dictionary.
    """
    labels = fs.get("config").get("labels") or config.get("labels")
    algorithms = fs.get("config").get("functions") or fs.get("config").get("algorithms") or config.get("algorithms")
    train_features = fs.get("config").get("columns") or fs.get("config").get("features") or config.get("train_features")

    train_df = df[train_features]
    features = []
    scores = {}
    out_df = pd.DataFrame(index=train_df.index)

    for label in labels:
        for model_config in algorithms:
            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = f"{label}_{algo_name}"
            model_pair = models.get(score_column_name)

            print(f"Predicting '{score_column_name}' with algorithm {algo_name}.")

            if algo_type == "gb":
                df_y_hat = predict_gb(model_pair, train_df, model_config)
            elif algo_type == "nn":
                df_y_hat = predict_nn(model_pair, train_df, model_config)
            elif algo_type == "lc":
                df_y_hat = predict_lc(model_pair, train_df, model_config)
            elif algo_type == "svc":
                df_y_hat = predict_svc(model_pair, train_df, model_config)
            else:
                raise ValueError(f"Unknown algorithm type '{algo_type}'")

            out_df[score_column_name] = df_y_hat
            features.append(score_column_name)

            if label in df:
                scores[score_column_name] = compute_scores(df[label], df_y_hat)

    return out_df, features, scores


def train_feature_set(df, fs, config):
    """
    Train models for the specified feature set and algorithms, then return predictions, models, and scores.
    """
    labels = fs.get("config").get("labels") or config.get("labels")
    algorithms = fs.get("config").get("functions") or fs.get("config").get("algorithms") or config.get("algorithms")
    train_features = fs.get("config").get("columns") or fs.get("config").get("features") or config.get("train_features")

    models = {}
    scores = {}
    out_df = pd.DataFrame()

    for label in labels:
        for model_config in algorithms:
            algo_name = model_config.get("name")
            algo_type = model_config.get("algo")
            score_column_name = f"{label}_{algo_name}"
            algo_train_length = model_config.get("train", {}).get("length")

            train_df = df.tail(algo_train_length) if algo_train_length else df
            df_X = train_df[train_features]
            df_y = train_df[label]

            print(f"Training '{score_column_name}' with algorithm {algo_name}.")

            if algo_type == "gb":
                model_pair = train_gb(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_gb(model_pair, df_X, model_config)
            elif algo_type == "nn":
                model_pair = train_nn(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_nn(model_pair, df_X, model_config)
            elif algo_type == "lc":
                model_pair = train_lc(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_lc(model_pair, df_X, model_config)
            elif algo_type == "svc":
                model_pair = train_svc(df_X, df_y, model_config)
                models[score_column_name] = model_pair
                df_y_hat = predict_svc(model_pair, df_X, model_config)
            else:
                print(f"ERROR: Unknown algorithm type '{algo_type}'.")
                return

            scores[score_column_name] = compute_scores(df_y, df_y_hat)
            out_df[score_column_name] = df_y_hat

    return out_df, models, scores


def resolve_generator_name(gen_name: str):
    """
    Resolve a generator function name to its function reference from a module.
    """
    mod_and_func = gen_name.split(':', 1)
    mod_name = mod_and_func[0] if len(mod_and_func) > 1 else None
    func_name = mod_and_func[-1]

    if not mod_name:
        return None

    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return None

    try:
        func = getattr(mod, func_name)
    except AttributeError:
        return None

    return func
