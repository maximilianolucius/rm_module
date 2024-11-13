from typing import Union, List

import numpy as np
import pandas as pd
from scipy import stats


def add_past_weighted_aggregations(df, column_name: str, weight_column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    """Add past weighted aggregations of a column based on specified weights and aggregation function."""
    return _add_weighted_aggregations(df, False, column_name, weight_column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def add_past_aggregations(df, column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    """Add past aggregations for a column using a specified aggregation function."""
    return _add_aggregations(df, False, column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def add_future_aggregations(df, column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    """Add future aggregations for a column using a specified aggregation function."""
    return _add_aggregations(df, True, column_name, fn, windows, suffix, rel_column_name, rel_factor, last_rows)


def _add_aggregations(df, is_future: bool, column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    """
    Compute moving aggregations for past or future values of a specified column over defined windows.
    The result columns are added to the DataFrame, and the column names are returned.
    """
    column = df[column_name]
    windows = [windows] if isinstance(windows, int) else windows
    suffix = suffix or f"_{fn.__name__}"

    if rel_column_name:
        rel_column = df[rel_column_name]

    features = []
    for w in windows:
        feature = _aggregate_last_rows(column, w, last_rows, fn) if last_rows else column.rolling(window=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
        feature = feature.shift(-w) if is_future else feature

        feature_name = f"{column_name}{suffix}_{w}"
        features.append(feature_name)
        df[feature_name] = rel_factor * (feature - rel_column) / rel_column if rel_column_name else rel_factor * feature

    return features


def _add_weighted_aggregations(df, is_future: bool, column_name: str, weight_column_name: str, fn, windows: Union[int, List[int]], suffix=None, rel_column_name: str = None, rel_factor: float = 1.0, last_rows: int = 0):
    """
    Compute weighted aggregations for past or future values of a specified column over defined windows.
    """
    column = df[column_name]
    weight_column = df[weight_column_name] if weight_column_name else pd.Series(data=1.0, index=column.index)
    products_column = column * weight_column
    windows = [windows] if isinstance(windows, int) else windows
    suffix = suffix or f"_{fn.__name__}"

    if rel_column_name:
        rel_column = df[rel_column_name]

    features = []
    for w in windows:
        feature = _aggregate_last_rows(products_column, w, last_rows, fn) if last_rows else products_column.rolling(window=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
        weights = _aggregate_last_rows(weight_column, w, last_rows, fn) if last_rows else weight_column.rolling(window=w, min_periods=max(1, w // 2)).apply(fn, raw=True)
        feature = feature / weights
        feature = feature.shift(-w) if is_future else feature

        feature_name = f"{column_name}{suffix}_{w}"
        features.append(feature_name)
        df[feature_name] = rel_factor * (feature - rel_column) / rel_column if rel_column_name else rel_factor * feature

    return features


def add_area_ratio(df, is_future: bool, column_name: str, windows: Union[int, List[int]], suffix=None, last_rows: int = 0):
    """
    Compute area ratio of a column for past or future values over defined windows and add to DataFrame.
    """
    column = df[column_name]
    windows = [windows] if isinstance(windows, int) else windows
    suffix = suffix or "_area_ratio"

    features = []
    for w in windows:
        ro = column.rolling(window=w, min_periods=max(1, w // 2))
        feature = ro.apply(area_fn, kwargs=dict(is_future=is_future), raw=True) if not last_rows else _aggregate_last_rows(column, w, last_rows, area_fn, is_future)
        feature_name = f"{column_name}{suffix}_{w}"
        df[feature_name] = feature.shift(-(w-1)) if is_future else feature
        features.append(feature_name)

    return features


def area_fn(x, is_future):
    """Compute area ratio for a series relative to the first or last element, scaled to [-1, +1]."""
    level = x[0] if is_future else x[-1]
    x_diff = x - level
    pos = (np.nansum(np.absolute(x_diff)) + np.nansum(x_diff)) / 2
    return (pos / np.nansum(np.absolute(x_diff)) * 2) - 1


def add_linear_trends(df, is_future: bool, column_name: str, windows: Union[int, List[int]], suffix=None, last_rows: int = 0):
    """
    Compute linear trend (slope) for a column for past or future values over defined windows and add to DataFrame.
    """
    column = df[column_name]
    windows = [windows] if isinstance(windows, int) else windows
    suffix = suffix or "_trend"

    features = []
    for w in windows:
        ro = column.rolling(window=w, min_periods=max(1, w // 2))
        feature = ro.apply(slope_fn, raw=True) if not last_rows else _aggregate_last_rows(column, w, last_rows, slope_fn)
        feature_name = f"{column_name}{suffix}_{w}"
        df[feature_name] = feature.shift(-(w-1)) if is_future else feature
        features.append(feature_name)

    return features


def slope_fn(x):
    """Calculate slope of a fitted linear regression line for a given series."""
    X_array = np.arange(len(x))
    y_array = x
    if np.isnan(y_array).any():
        nans = ~np.isnan(y_array)
        X_array = X_array[nans]
        y_array = y_array[nans]

    slope, intercept, r, p, se = stats.linregress(X_array, y_array)
    return slope


def to_log_diff(sr):
    """Convert a series to log differences."""
    return np.log(sr).diff()


def to_diff_NEW(sr):
    """Convert a series to percentage differences."""
    return 100 * sr.diff() / sr


def to_diff(sr):
    """
    Calculate differences for a series as a percentage.
    Each value is the difference between the current and previous values divided by the current value.
    """
    return sr.rolling(window=2, min_periods=2).apply(lambda x: 100 * (x[1] - x[0]) / x[0], raw=True)


def _aggregate_last_rows(column, window, last_rows, fn, *args):
    """
    Apply a rolling aggregation for only the last n rows.
    """
    length = len(column)
    values = [fn(column.iloc[-window - r:length - r].to_numpy(), *args) for r in range(last_rows)]
    feature = pd.Series(data=np.nan, index=column.index, dtype=float)
    feature.iloc[-last_rows:] = list(reversed(values))
    return feature
