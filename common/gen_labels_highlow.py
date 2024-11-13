import numpy as np

from common.gen_features_rolling_agg import *


def generate_labels_highlow(df, horizon):
    """
    Generate (compute) a number of labels similar to other derived features but using future data.
    This function is used before training to generate true labels.
    """
    labels = []
    windows = [horizon]

    # Max high for horizon relative to close (normally positive but can be negative)
    labels += add_future_aggregations(df, "high", np.max, windows=windows, suffix='_max', rel_column_name="close", rel_factor=100.0)
    high_column_name = "high_max_" + str(horizon)  # Example: high_max_180

    # Max high crosses the threshold
    labels += add_threshold_feature(df, high_column_name, thresholds=[1.0, 1.5, 2.0, 2.5, 3.0], out_names=["high_10", "high_15", "high_20", "high_25", "high_30"])
    # Max high does not cross the threshold
    labels += add_threshold_feature(df, high_column_name, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], out_names=["high_01", "high_02", "high_03", "high_04", "high_05"])

    # Min low for horizon relative to close (normally negative but can be positive)
    labels += add_future_aggregations(df, "low", np.min, windows=windows, suffix='_min', rel_column_name="close", rel_factor=100.0)
    low_column_name = "low_min_" + str(horizon)  # Example: low_min_180

    # Min low does not cross the negative threshold
    labels += add_threshold_feature(df, low_column_name, thresholds=[-0.1, -0.2, -0.3, -0.4, -0.5], out_names=["low_01", "low_02", "low_03", "low_04", "low_05"])
    # Min low crosses the negative threshold
    labels += add_threshold_feature(df, low_column_name, thresholds=[-1.0, -1.5, -2.0, -2.5, -3.0], out_names=["low_10", "low_15", "low_20", "low_25", "low_30"])

    # Ratio high_to_low_window
    df[high_column_name] = df[high_column_name].clip(lower=0)
    df[low_column_name] = df[low_column_name].clip(upper=0)
    df[low_column_name] = df[low_column_name] * -1
    column_sum = df[high_column_name] + df[low_column_name]
    ratio_column_name = "high_to_low_" + str(horizon)
    ratio_column = df[high_column_name] / column_sum  # in [0,1]
    df[ratio_column_name] = (ratio_column * 2) - 1

    return labels


def generate_labels_highlow2(df, config: dict):
    """
    Generate multiple increase/decrease labels which are typically used for training.
    """
    column_names = config.get('columns')
    close_column = column_names[0]
    high_column = column_names[1]
    low_column = column_names[2]

    function = config.get('function')
    if function not in ['high', 'low']:
        raise ValueError(f"Unknown function name {function}. Only 'high' or 'low' are possible")

    tolerance = config.get('tolerance')
    thresholds = config.get('thresholds')
    if not isinstance(thresholds, list):
        thresholds = [thresholds]

    if function == 'high':
        thresholds = [abs(t) for t in thresholds]
        price_columns = [high_column, low_column]
    elif function == 'low':
        thresholds = [-abs(t) for t in thresholds]
        price_columns = [low_column, high_column]

    tolerances = [round(-t*tolerance, 6) for t in thresholds]
    horizon = config.get('horizon')
    names = config.get('names')
    if len(names) != len(thresholds):
        raise ValueError(f"'highlow2' Label generator: for each threshold value one name has to be provided.")

    labels = []
    for i, threshold in enumerate(thresholds):
        first_cross_labels(df, horizon, [threshold, tolerances[i]], close_column, price_columns, names[i])
        labels.append(names[i])

    print(f"Highlow2 labels computed: {labels}")

    return df, labels


def generate_labels_sim(df, horizon):
    """Currently not used."""
    labels = []

    add_future_aggregations(df, "high", np.max, horizon, suffix='_max', rel_column_name="close", rel_factor=100.0)
    labels += add_threshold_feature(df, "high_max_180", thresholds=[2.0], out_names=["high_20"])
    labels += add_threshold_feature(df, "high_max_180", thresholds=[0.2], out_names=["high_02"])
    add_future_aggregations(df, "low", np.min, horizon, suffix='_min', rel_column_name="close", rel_factor=100.0)
    labels += add_threshold_feature(df, "low_min_180", thresholds=[-0.2], out_names=["low_02"])
    labels += add_threshold_feature(df, "low_min_180", thresholds=[-2.0], out_names=["low_20"])

    return labels


def generate_labels_regressor(df, horizon):
    """Labels for regression. Currently not used."""
    labels = []

    labels += add_future_aggregations(df, "high", np.max, horizon, suffix='_max', rel_column_name="close", rel_factor=100.0)
    labels += add_future_aggregations(df, "low", np.min, horizon, suffix='_min', rel_column_name="close", rel_factor=100.0)

    return labels


def _first_location_of_crossing_threshold(df, horizon, threshold, close_column_name, price_column_name):
    """
    First location of crossing the threshold.
    For each point, take its close price, and then find the distance (location, index)
    to the first future point with high or low price higher or lower, respectively
    than the close price.
    """
    def fn_high(x):
        if len(x) < 2:
            return np.nan
        p = x[0, 0]
        p_threshold = p * (1 + (threshold / 100.0))
        idx = np.argmax(x[1:, 1] > p_threshold)

        if idx == 0 and x[1, 1] <= p_threshold:
            return np.nan
        return idx

    def fn_low(x):
        if len(x) < 2:
            return np.nan
        p = x[0, 0]
        p_threshold = p * (1 + (threshold / 100.0))
        idx = np.argmax(x[1:, 1] < p_threshold)

        if idx == 0 and x[1, 1] >= p_threshold:
            return np.nan
        return idx

    rl = df[[close_column_name, price_column_name]].rolling(horizon + 1, min_periods=(horizon // 2), method='table')

    if threshold > 0:
        df_out = rl.apply(fn_high, raw=True, engine='numba')
    elif threshold < 0:
        df_out = rl.apply(fn_low, raw=True, engine='numba')
    else:
        raise ValueError(f"Threshold cannot be zero.")

    df_out = df_out.shift(-horizon)
    out_column = df_out.iloc[:, 0]

    return out_column


def first_cross_labels(df, horizon, thresholds, close_column, price_columns, out_column):
    """
    Produce one boolean column which is true if the price crosses the first threshold
    but does not cross the second threshold in the opposite direction before that.
    """
    df["first_idx_column"] = _first_location_of_crossing_threshold(df, horizon, thresholds[0], close_column, price_columns[0])
    df["second_idx_column"] = _first_location_of_crossing_threshold(df, horizon, thresholds[1], close_column, price_columns[1])

    def is_high_true(x):
        if np.isnan(x[0]):
            return False
        elif np.isnan(x[1]):
            return True
        else:
            return x[0] <= x[1]

    df[out_column] = df[["first_idx_column", "second_idx_column"]].apply(is_high_true, raw=True, axis=1)
    df.drop(columns=['first_idx_column', 'second_idx_column'], inplace=True)

    return out_column


if __name__ == "__main__":
    pass
