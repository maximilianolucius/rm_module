import dateparser
import pytz
import json
from decimal import *

import numpy as np
import pandas as pd

from apscheduler.triggers.cron import CronTrigger
from common.gen_features import *


#
# Decimals
#

def to_decimal(value):
    """Convert value to a decimal with required precision.

    Handles string, float, or decimal inputs and quantizes to a fixed precision.
    """
    n = 8
    rr = Decimal(1) / (Decimal(10) ** n)
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_DOWN)
    return ret


def round_str(value, digits):
    """Round value to a string with specified digits, rounding half up."""
    rr = Decimal(1) / (Decimal(10) ** digits)
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_HALF_UP)
    return f"{ret:.{digits}f}"


def round_down_str(value, digits):
    """Round value down to specified digits as a string."""
    rr = Decimal(1) / (Decimal(10) ** digits)
    ret = Decimal(str(value)).quantize(rr, rounding=ROUND_DOWN)
    return f"{ret:.{digits}f}"


#
# Binance-specific Utilities
#

def klines_to_df(klines, df):
    """Convert Binance klines to a DataFrame and append to an existing DataFrame if provided."""
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    dtypes = {
        'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64',
        'volume': 'float64', 'close_time': 'int64', 'quote_av': 'float64',
        'trades': 'int64', 'tb_base_av': 'float64', 'tb_quote_av': 'float64',
        'ignore': 'float64',
    }
    data = data.astype(dtypes)

    df = pd.concat([df, data]) if df is not None else data
    df = df.drop_duplicates(subset=["timestamp"]).set_index('timestamp')
    return df


def binance_klines_to_df(klines: list):
    """Convert a list of Binance klines to a DataFrame with appropriate data types and index."""
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
    ]
    df = pd.DataFrame(klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    num_columns = ["open", "high", "low", "close", "volume", "quote_av", "trades", "tb_base_av", "tb_quote_av"]
    df[num_columns] = df[num_columns].apply(pd.to_numeric)

    if "timestamp" in df.columns:
        df.set_index('timestamp', inplace=True)

    return df


def binance_freq_from_pandas(freq: str) -> str:
    """Convert pandas frequency to Binance API compatible frequency."""
    freq = freq.replace("min", "m").replace("D", "d").replace("W", "w").replace("BMS", "M")
    freq = "1" + freq if len(freq) == 1 else freq
    if not (2 <= len(freq) <= 3) or not freq[:-1].isdigit() or freq[-1] not in ["m", "h", "d", "w", "M"]:
        raise ValueError(f"Unsupported Binance frequency {freq}")
    return freq


def binance_get_interval(freq: str, timestamp: int = None):
    """Get start and end interval in milliseconds for a timestamp according to Binance frequency."""
    timestamp = timestamp or datetime.utcnow()
    timestamp = pd.to_datetime(timestamp, unit='ms').to_pydatetime() if isinstance(timestamp, int) else timestamp
    timestamp = timestamp.replace(microsecond=0, tzinfo=timezone.utc)

    if freq == "1s":
        start = timestamp.timestamp()
        end = (timestamp + timedelta(seconds=1)).timestamp()
    elif freq == "5s":
        ref_ts = timestamp.replace(second=0)
        intervals = (timestamp - ref_ts).total_seconds() // 5
        start = ref_ts + timedelta(seconds=5 * intervals)
        end = start + timedelta(seconds=5)
    elif freq == "1m":
        start = timestamp.replace(second=0).timestamp()
        end = (timestamp + timedelta(minutes=1)).timestamp()
    elif freq == "1h":
        start = timestamp.replace(minute=0, second=0).timestamp()
        end = (timestamp + timedelta(hours=1)).timestamp()
    else:
        raise ValueError("Unsupported frequency")

    return int(start * 1000), int(end * 1000)


def pandas_get_interval(freq: str, timestamp: int = None):
    """Get interval start and end for pandas frequency based on a timestamp."""
    timestamp = timestamp or int(datetime.now(pytz.utc).timestamp())
    timestamp = int(timestamp.replace(tzinfo=pytz.utc).timestamp()) if isinstance(timestamp, datetime) else timestamp

    interval_length_sec = pandas_interval_length_ms(freq) / 1000
    start = (timestamp // interval_length_sec) * interval_length_sec
    end = start + interval_length_sec

    return int(start * 1000), int(end * 1000)


def pandas_interval_length_ms(freq: str):
    """Return length of pandas interval in milliseconds."""
    return int(pd.Timedelta(freq).total_seconds() * 1000)


#
# Date and Time Utilities
#

def freq_to_CronTrigger(freq: str):
    """Convert frequency string to a CronTrigger for APScheduler."""
    if freq.endswith("min"):
        trigger = CronTrigger(minute="*/" + freq[:-3], second="1", timezone="UTC") if freq[:-3] != "1" else CronTrigger(
            minute="*", second="1", timezone="UTC")
    elif freq.endswith("h"):
        trigger = CronTrigger(hour="*/" + freq[:-1], minute="0", second="2", timezone="UTC") if freq[
                                                                                                :-1] != "1" else CronTrigger(
            hour="*", minute="0", second="2", timezone="UTC")
    elif freq.endswith("D"):
        trigger = CronTrigger(day="*/" + freq[:-1], second="5", timezone="UTC") if freq[:-1] != "1" else CronTrigger(
            day="*", second="5", timezone="UTC")
    elif freq.endswith("W"):
        trigger = CronTrigger(day="*/" + freq[:-1], second="10", timezone="UTC") if freq[:-1] != "1" else CronTrigger(
            week="*", second="10", timezone="UTC")
    elif freq.endswith("MS"):
        trigger = CronTrigger(month="*/" + freq[:-2], second="30", timezone="UTC") if freq[:-2] != "1" else CronTrigger(
            month="*", second="30", timezone="UTC")
    else:
        raise ValueError(f"Cannot convert frequency '{freq}' to cron.")

    return trigger


def now_timestamp():
    """Get current UTC timestamp in milliseconds."""
    return int(datetime.utcnow().replace(tzinfo=timezone.utc).timestamp() * 1000)


def find_index(df: pd.DataFrame, date_str: str, column_name: str = "timestamp"):
    """Return index of the record with the specified datetime string in a column."""
    d = dateparser.parse(date_str)
    d = d.replace(tzinfo=pytz.utc) if d.tzinfo is None else d.replace(tzinfo=None)
    res = df[df[column_name] == d]

    if res.empty:
        raise ValueError(f"Cannot find date '{date_str}' in the column '{column_name}'.")

    return res.index[0]


def notnull_tail_rows(df):
    """Get the maximum number of tail rows without nulls."""
    nan_df = df.isnull()
    nan_cols = nan_df.any()
    if not nan_cols.any():
        return len(df)

    tail_rows = nan_df[nan_cols[nan_cols].index].values[::-1].argmax(axis=0).min()
    return tail_rows
