import os
import sys
import importlib
from datetime import datetime, timezone, timedelta
from typing import Union
import json
import itertools

import numpy as np
import pandas as pd

import scipy.stats as stats

from common.utils import *
from common.gen_features_rolling_agg import *
from common.gen_features_rolling_agg import _aggregate_last_rows

"""
Feature generators.
A feature generator knows how to generate features from its declarative specification in the config file.
"""


def generate_features_tsfresh(df, config: dict, last_rows: int = 0):
    """
    This feature generator relies on tsfresh functions.
    tsfresh depends on matrixprofile for which binaries are not available for many versions.
    Therefore, the use of tsfresh may require Python 3.8
    """
    # Imported here to avoid installation of tsfresh if not used
    import tsfresh.feature_extraction.feature_calculators as tsf

    # Transform str/list and list to dict with argument names as keys and column names as values
    column_names = config.get('columns')
    if not column_names:
        raise ValueError(f"No input column for feature generator 'stats': {column_names}")

    if isinstance(column_names, str):
        column_name = column_names
    elif isinstance(column_names, list):
        column_name = column_names[0]
    elif isinstance(column_names, dict):
        column_name = next(iter(column_names.values()))
    else:
        raise ValueError(f"Columns should be a string, list or dict. Wrong type: {type(column_names)}")

    column = df[column_name].interpolate()

    windows = config.get('windows')
    if not isinstance(windows, list):
        windows = [windows]

    features = []
    for w in windows:
        ro = column.rolling(window=w, min_periods=max(1, w // 2))

        # Skewness feature
        feature_name = column_name + "_skewness_" + str(w)
        if not last_rows:
            df[feature_name] = ro.apply(tsf.skewness, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.skewness)
        features.append(feature_name)

        # Kurtosis feature
        feature_name = column_name + "_kurtosis_" + str(w)
        if not last_rows:
            df[feature_name] = ro.apply(tsf.kurtosis, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.kurtosis)
        features.append(feature_name)

        # Mean second derivative central feature
        feature_name = column_name + "_msdc_" + str(w)
        if not last_rows:
            df[feature_name] = ro.apply(tsf.mean_second_derivative_central, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.mean_second_derivative_central)
        features.append(feature_name)

        # Longest strike below mean feature
        feature_name = column_name + "_lsbm_" + str(w)
        if not last_rows:
            df[feature_name] = ro.apply(tsf.longest_strike_below_mean, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.longest_strike_below_mean)
        features.append(feature_name)

        # First location of maximum feature
        feature_name = column_name + "_fmax_" + str(w)
        if not last_rows:
            df[feature_name] = ro.apply(tsf.first_location_of_maximum, raw=True)
        else:
            df[feature_name] = _aggregate_last_rows(column, w, last_rows, tsf.first_location_of_maximum)
        features.append(feature_name)

    return features


def generate_features_talib(df, config: dict, last_rows: int = 0):
    """
    Apply TA functions from talib according to the specified configuration parameters.
    TA-lib is sensitive to NaN values, so NaNs should be removed for meaningful results.
    """
    rel_base = config.get('parameters', {}).get('rel_base', False)
    rel_func = config.get('parameters', {}).get('rel_func', False)
    percentage = config.get('parameters', {}).get('percentage', False)
    log = config.get('parameters', {}).get('log', False)

    # Import talib modules for accessing TA functions
    mod_name = "talib"
    talib_mod = sys.modules.get(mod_name)
    if talib_mod is None:
        try:
            talib_mod = importlib.import_module(mod_name)
        except Exception as e:
            raise ValueError(f"Cannot import module {mod_name}. Check if talib is installed correctly")

    mod_name = "talib.stream"
    talib_mod_stream = sys.modules.get(mod_name)
    if talib_mod_stream is None:
        try:
            talib_mod_stream = importlib.import_module(mod_name)
        except Exception as e:
            raise ValueError(f"Cannot import module {mod_name}. Check if talib is installed correctly")

    mod_name = "talib.abstract"
    talib_mod_abstract = sys.modules.get(mod_name)
    if talib_mod_abstract is None:
        try:
            talib_mod_abstract = importlib.import_module(mod_name)
        except Exception as e:
            raise ValueError(f"Cannot import module {mod_name}. Check if talib is installed correctly")

    # Process configuration parameters
    column_names = config.get('columns')
    if isinstance(column_names, str):
        column_names = {'real': column_names}
    elif isinstance(column_names, list) and len(column_names) == 1:
        column_names = {'real': column_names[0]}
    elif isinstance(column_names, list):
        column_names = {f'real{i}': col for i, col in enumerate(column_names)}
    elif isinstance(column_names, dict):
        pass
    else:
        raise ValueError(f"Columns should be a string, list or dict. Wrong type: {type(column_names)}")

    columns = {arg: df[col_name].interpolate() for arg, col_name in column_names.items()}
    col_out_names = "_".join(column_names.values())

    func_names = config.get('functions')
    if not isinstance(func_names, list):
        func_names = [func_names]

    windows = config.get('windows')
    if not isinstance(windows, list):
        windows = [windows]

    names = config.get('names')

    outs = []
    features = []
    for func_name in func_names:
        fn_outs = []
        fn_out_names = []

        try:
            fn = getattr(talib_mod_abstract, func_name)
        except AttributeError as e:
            raise ValueError(f"Cannot resolve talib function name '{func_name}'. Check the name of the function")
        is_streamable_function = fn.function_flags is None or 'Function has an unstable period' not in fn.function_flags
        is_streamable_function = False

        for j, w in enumerate(windows):
            if not last_rows or not w or not is_streamable_function:
                try:
                    fn = getattr(talib_mod, func_name)
                except AttributeError as e:
                    raise ValueError(f"Cannot resolve talib function name '{func_name}'. Check the name of the function")

                args = columns.copy()
                if w:
                    args['timeperiod'] = w
                if w == 1 and len(columns) == 1:
                    out = next(iter(columns.values()))
                else:
                    out = fn(**args)

            else:
                try:
                    fn = getattr(talib_mod_stream, func_name)
                except AttributeError as e:
                    raise ValueError(f"Cannot resolve talib.stream function name '{func_name}'. Check the name of the function")

                out_values = []
                for r in range(last_rows):
                    args = {k: v.iloc[:len(v)-r] for k, v in columns.items()}
                    if w:
                        args['timeperiod'] = w

                    if w == 1 and len(columns) == 1:
                        col = next(iter(columns.values()))
                        out_val = col.iloc[-r-1]
                    else:
                        out_val = fn(**args)
                    out_values.append(out_val)

                out = pd.Series(data=np.nan, index=df.index, dtype=float)
                out.iloc[-last_rows:] = list(reversed(out_values))

            if not w:
                if not names:
                    out_name = f"{col_out_names}_{func_name}"
                elif isinstance(names, str):
                    out_name = names
                elif isinstance(names, list):
                    out_name = names[j]
            else:
                out_name = f"{col_out_names}_{func_name}_"
                win_name = str(w)
                if not names:
                    out_name = out_name + win_name
                elif isinstance(names, str):
                    out_name = out_name + names + "_" + win_name
                elif isinstance(names, list):
                    out_name = out_name + names[j]

            fn_out_names.append(out_name)
            out.name = out_name
            fn_outs.append(out)

        fn_outs = _convert_to_relative(fn_outs, rel_base, rel_func, percentage)
        features.extend(fn_out_names)
        outs.extend(fn_outs)

    for out in outs:
        df[out.name] = np.log(out) if log else out

    return features


def _convert_to_relative(fn_outs: list, rel_base, rel_func, percentage):
    rel_outs = []
    size = len(fn_outs)
    for i, feature in enumerate(fn_outs):
        if not rel_base:
            rel_out = fn_outs[i]
        elif (rel_base == "next" or rel_base == "last") and i == size - 1:
            rel_out = fn_outs[i]
        elif (rel_base == "prev" or rel_base == "first") and i == 0:
            rel_out = fn_outs[i]

        elif rel_base == "next" or rel_base == "last":
            base = fn_outs[i + 1] if rel_base == "next" else fn_outs[size - 1]
            if rel_func == "rel":
                rel_out = feature / base
            elif rel_func == "diff":
                rel_out = feature - base
            elif rel_func == "rel_diff":
                rel_out = (feature - base) / base
            else:
                raise ValueError(f"Unknown value of the 'rel_func' config parameter: {rel_func=}")

        elif rel_base == "prev" or rel_base == "first":
            base = fn_outs[i - 1] if rel_base == "prev" else fn_outs[0]
            if rel_func == "rel":
                rel_out = feature / base
            elif rel_func == "diff":
                rel_out = feature - base
            elif rel_func == "rel_diff":
                rel_out = (feature - base) / base
            else:
                raise ValueError(f"Unknown value of the 'rel_func' config parameter: {rel_func=}")

        if percentage:
            rel_out = rel_out * 100.0

        rel_out.name = fn_outs[i].name
        rel_outs.append(rel_out)

    return rel_outs


def fmax_fn(x):
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


def lsbm_fn(x):
    """
    The longest consecutive interval of values higher than the mean.
    Equivalent of tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean
    """

    def _get_length_sequences_where(x):
        return [len(list(group)) for value, group in itertools.groupby(x) if value == 1] or [0]

    return np.max(_get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0


if __name__ == "__main__":
    pass
