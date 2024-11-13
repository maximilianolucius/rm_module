import queue
import logging
import queue

import numpy as np
import pandas as pd

from common.generators import predict_feature_set
from common.model_store import *
from common.utils import *
from scripts.features import *
from scripts.merge import *

log = logging.getLogger('analyzer')


class Analyzer:
    """
    In-memory database representing the current state of the trading environment, including history.
    """

    def __init__(self, config):
        """
        Initialize Analyzer with config defining database properties, model paths, and data structures.
        """
        self.config = config
        self.klines = {}  # Stores recent klines data
        self.queue = queue.Queue()

        # Load models
        symbol = App.config["symbol"]
        data_path = Path(App.config["data_folder"]) / symbol
        model_path = Path(App.config["model_folder"])
        model_path = data_path / model_path if not model_path.is_absolute() else model_path
        self.models = load_models(model_path.resolve(), App.config["labels"], App.config["algorithms"])

        # Load latest transaction and simulated trade state
        App.transaction = load_last_transaction()

    #
    # Data Operations
    #
    def get_klines_count(self, symbol):
        return len(self.klines.get(symbol, []))

    def get_last_kline(self, symbol):
        return self.klines.get(symbol)[-1] if self.get_klines_count(symbol) > 0 else None

    def get_last_kline_ts(self, symbol):
        last_kline = self.get_last_kline(symbol)
        return last_kline[0] if last_kline else 0

    def get_missing_klines_count(self, symbol):
        last_kline_ts = self.get_last_kline_ts(symbol)
        if not last_kline_ts:
            return App.config["features_horizon"]

        freq = App.config["freq"]
        now = datetime.utcnow()
        last_kline = datetime.utcfromtimestamp(last_kline_ts // 1000)
        interval_length = pd.Timedelta(freq).to_pytimedelta()
        intervals_count = (now - last_kline) // interval_length + 2
        return intervals_count

    def store_klines(self, data: dict):
        """
        Store latest klines for symbols, overwriting any existing data with the same timestamp.
        """
        now_ts = now_timestamp()
        interval_length_ms = pandas_interval_length_ms(App.config["freq"])

        for symbol, klines in data.items():
            klines_data = self.klines.setdefault(symbol, [])
            ts = klines[0][0]

            # Remove existing overlap and append new klines
            existing_indexes = [i for i, x in enumerate(klines_data) if x[0] >= ts]
            if existing_indexes:
                del klines_data[min(existing_indexes):]
            klines_data.extend(klines)

            # Remove older data beyond the configured horizon
            to_delete = len(klines_data) - App.config["features_horizon"]
            if to_delete > 0:
                del klines_data[:to_delete]

            # Validate consistency in timestamps
            for i, kline in enumerate(self.klines[symbol]):
                if i > 0 and kline[0] - prev_ts != interval_length_ms:
                    log.error("Klines are expected to be in a regular 1m time series.")
                prev_ts = kline[0]

            log.debug(
                f"Stored klines. Last kline end: {self.get_last_kline_ts(symbol) + interval_length_ms}, Current time: {now_ts}")

    def store_depth(self, depths: list, freq):
        """
        Store order book data to files. Each entry in depths represents a response from the order book request.
        """
        TRADE_DATA = "."  # Default data directory; can be customized
        for depth in depths:
            symbol = depth["symbol"]
            path = Path(TRADE_DATA, App.config["collector"]["folder"], App.config["collector"]["depth"]["folder"])
            path.mkdir(parents=True, exist_ok=True)
            file = path / f"depth-{symbol}-{freq}.txt"
            with open(file, 'a+') as f:
                f.write(json.dumps(depth) + "\n")

    def store_queue(self):
        """
        Store queued data events persistently, organized by type, symbol, and frequency.
        """
        events = {}
        while True:
            try:
                item = self.queue.get_nowait()
                if item:
                    events.setdefault(item.get("e"), {}).setdefault(item.get("s"), []).append(item)
                self.queue.task_done()
            except queue.Empty:
                break

        path = Path(".", App.config["collector"]["folder"], App.config["collector"]["stream"]["folder"])
        path.mkdir(parents=True, exist_ok=True)

        for c, symbols in events.items():
            for s, data in symbols.items():
                file_name = f"{c}-{s}-{datetime.utcnow():%Y%m}.txt"
                file = path / file_name
                with open(file, 'a+') as f:
                    f.write("\n".join(json.dumps(event) for event in data) + "\n")

    #
    # Data Analysis
    #
    def analyze(self, ignore_last_rows=False):
        """
        Analyze the most recent data to generate derived features, predictions, and trade signals.
        """
        symbol = App.config["symbol"]
        last_rows = App.config["features_last_rows"]

        log.info(
            f"Analyzing {symbol}. Last kline timestamp: {pd.to_datetime(self.get_last_kline_ts(symbol), unit='ms')}")

        # Convert klines to DataFrames and merge sources
        data_sources = App.config.get("data_sources", [])
        for ds in data_sources:
            if ds.get("file") == "klines":
                try:
                    klines = self.klines.get(ds["folder"])
                    df = binance_klines_to_df(klines)
                    if df.isnull().any().any():
                        log.warning(f"Null values found in source data: {df.isnull().sum().to_dict()}")
                except Exception as e:
                    log.error(f"Error in klines_to_df: {e}. Length of klines: {len(klines)}")
                    return
            else:
                log.error("Only 'klines' is supported as a data source in 'data_sources' config.")
                return
            ds["df"] = df

        df = merge_data_sources(data_sources)

        # Generate derived features
        feature_sets = App.config.get("feature_sets", [])
        feature_columns = []
        for fs in feature_sets:
            df, feats = generate_feature_set(df, fs, last_rows=last_rows if not ignore_last_rows else 0)
            feature_columns.extend(feats)

        # Restrict data to necessary rows for predictions
        if not ignore_last_rows:
            df = df.iloc[-last_rows:]
        df = df.tail(notnull_tail_rows(df[App.config["train_features"]]))

        # Generate predictions for each model
        predict_df = df[App.config["train_features"]]
        if predict_df.isnull().any().any():
            log.error(f"Null in predict_df: {predict_df.isnull().sum().to_dict()}")
            return

        train_feature_sets = App.config.get("train_feature_sets", [])
        if not train_feature_sets:
            log.error("No train feature sets defined in config.")
            return

        score_df = pd.DataFrame(index=predict_df.index)
        train_feature_columns = []
        for fs in train_feature_sets:
            fs_df, feats, _ = predict_feature_set(predict_df, fs, App.config, self.models)
            score_df = pd.concat([score_df, fs_df], axis=1)
            train_feature_columns.extend(feats)

        df = pd.concat([df, score_df], axis=1)

        # Generate trading signals
        signal_sets = App.config.get("signal_sets", [])
        if not signal_sets:
            log.error("No signal sets defined in config.")
            return

        signal_columns = []
        for fs in signal_sets:
            df, feats = generate_feature_set(df, fs, last_rows=last_rows if not ignore_last_rows else 0)
            signal_columns.extend(feats)

        # Log latest signal values
        row = df.iloc[-1]
        scores = ", ".join(
            [f"{x}={row[x]:+.3f}" if isinstance(row[x], float) else f"{x}={str(row[x])}" for x in signal_columns])
        log.info(f"Close: {int(row['close']):,}, Signals: {scores}")

        if App.df is None or App.df.empty:
            App.df = df
            return

        # Check consistency of last computed rows
        for r in range(2, 5):
            idx = df.index[-r - 1]
            if idx in App.df.index:
                old_row, new_row = App.df.loc[idx], df.loc[idx]
                if not np.all(np.isclose(old_row, new_row)):
                    log.warning(
                        f"Discrepancy for '{idx}'. NEW: {new_row[~np.isclose(old_row, new_row)].to_dict()}, OLD: {old_row[~np.isclose(old_row, new_row)].to_dict()}")

        # Append new data rows to main DataFrame
        App.df = df.tail(5).combine_first(App.df)
        if len(App.df) > App.config["features_horizon"] + 15:
            App.df = App.df.tail(App.config["features_horizon"])


if __name__ == "__main__":
    pass
