import pandas as pd


"""
Generate top and bottom label columns with specified parameters:
- level: defines the minimum "jump" from the minimum or maximum
- tolerance: defines the distance from minimum or maximum for the label to be true.

Concepts:

* Level (fraction): Specifies the minimum "jump" required up or down for an extremum to be selected.
The extremum must be surrounded by points lower/higher by this amount on both sides. The level is a fraction of the extremum value.
* Tolerance (fraction): Defines the width around an extremum. It selects points on either side that differ from the extremum by this fraction.

The labels, scores, and source columns are stored in an output file.
"""


def generate_labels_topbot2(df, config: dict):
    """Generate and label top or bottom points with specified configuration."""
    init_column_number = len(df.columns)

    column_name = config.get('columns')
    if not column_name or not isinstance(column_name, str) or column_name not in df.columns:
        raise ValueError("The 'columns' parameter must be a valid column name present in the DataFrame.")

    function = config.get('function')
    if function not in ['top', 'bot']:
        raise ValueError("Unknown function name. Only 'top' or 'bot' are allowed.")

    tolerances = config.get('tolerances')
    if not isinstance(tolerances, list):
        tolerances = [tolerances]

    level = config.get('level')
    level = abs(level) if function == 'top' else -abs(level)

    names = config.get('names')
    if len(names) != len(tolerances):
        raise ValueError("Each tolerance value requires a corresponding name.")

    labels = []
    for i, tolerance in enumerate(tolerances):
        df, new_labels = add_extremum_features(df, column_name, [level], abs(level) * tolerance, [names[i]])
        labels.extend(new_labels)

    labels = df.columns.to_list()[init_column_number:]
    return df, labels


def generate_labels_topbot(df, column_name: str, top_level_fracs: list, bot_level_fracs: list):
    """Generate top and bottom labels for various tolerance levels."""
    init_column_number = len(df.columns)
    tolerance_fracs = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03]

    for tolerance_frac in tolerance_fracs:
        top_labels = [f'top{i}_{int(tolerance_frac * 1000)}' for i in range(1, 6)]
        bot_labels = [f'bot{i}_{int(tolerance_frac * 1000)}' for i in range(1, 6)]

        df, labels = add_extremum_features(df, column_name, top_level_fracs, tolerance_frac, top_labels)
        print(f"Top labels computed: {top_labels}")
        df, labels = add_extremum_features(df, column_name, bot_level_fracs, tolerance_frac, bot_labels)
        print(f"Bottom labels computed: {bot_labels}")

    labels = df.columns.to_list()[init_column_number:]
    return df, labels


def add_extremum_features(df, column_name: str, level_fracs: list, tolerance_frac: float, out_names: list):
    """
    Add extremum label columns to the DataFrame for specified level fractions and tolerances.
    """
    column = df[column_name]
    out_columns = []
    for i, level_frac in enumerate(level_fracs):
        extrems = find_all_extremums(column, level_frac > 0, abs(level_frac), tolerance_frac)
        out_name = out_names[i]
        out_column = pd.Series(False, index=df.index, name=out_name)

        for extr in extrems:
            out_column.loc[extr[1]: extr[3]] = True

        out_columns.append(out_column)

    df = pd.concat([df] + out_columns, axis=1)
    return df, out_names


def find_all_extremums(sr: pd.Series, is_max: bool, level_frac: float, tolerance_frac: float) -> list:
    """
    Find all extremums (top or bottom) in the series based on level and tolerance intervals.
    """
    extremums = []
    intervals = [(sr.index[0], sr.index[-1] + 1)]

    while intervals:
        interval = intervals.pop()
        extremum = find_one_extremum(sr.loc[interval[0]: interval[1]], is_max, level_frac, tolerance_frac)

        if extremum[0] and extremum[-1]:
            extremums.append(extremum)

        if extremum[0] and interval[0] < extremum[0]:
            intervals.append((interval[0], extremum[0]))
        if extremum[-1] and extremum[-1] < interval[1]:
            intervals.append((extremum[-1], interval[1]))

    return sorted(extremums, key=lambda x: x[2])


def find_one_extremum(sr: pd.Series, is_max: bool, level_frac: float, tolerance_frac: float) -> tuple:
    """
    Find a single extremum (top or bottom) in the series and determine its level and tolerance intervals.
    """
    extr_idx = sr.idxmax() if is_max else sr.idxmin()
    extr_val = sr.loc[extr_idx]
    level_val = extr_val * (1 - level_frac) if is_max else extr_val / (1 - level_frac)
    tolerance_val = extr_val * (1 - tolerance_frac) if is_max else extr_val / (1 - tolerance_frac)

    sr_left = sr.loc[:extr_idx]
    sr_right = sr.loc[extr_idx:]

    left_level_idx = _left_level_idx(sr_left, is_max, level_val)
    right_level_idx = _right_level_idx(sr_right, is_max, level_val)

    left_tol_idx = _left_level_idx(sr_left, is_max, tolerance_val)
    right_tol_idx = _right_level_idx(sr_right, is_max, tolerance_val)

    return (left_level_idx, left_tol_idx, extr_idx, right_tol_idx, right_level_idx)


def _left_level_idx(sr_left: pd.Series, is_max: bool, level_val: float):
    """Find the left boundary index for level condition."""
    sr_left_level = sr_left[sr_left < level_val] if is_max else sr_left[sr_left > level_val]
    return sr_left_level.index[-1] if len(sr_left_level) > 0 else None


def _right_level_idx(sr_right: pd.Series, is_max: bool, level_val: float):
    """Find the right boundary index for level condition."""
    sr_right_level = sr_right[sr_right < level_val] if is_max else sr_right[sr_right > level_val]
    return sr_right_level.index[0] if len(sr_right_level) > 0 else None
