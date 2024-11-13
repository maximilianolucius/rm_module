import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).parent.parent


class App:
    """Globally accessible application variables and configurations."""

    loop = None  # Asyncio main loop
    sched = None  # Scheduler
    analyzer = None  # Store and analyze data
    client = None  # Connector client
    bm = None  # WebSocket manager for push notifications
    conn_key = None  # WebSocket connection key

    # System status indicators
    error_status = 0  # Network or connection issues
    server_status = 0  # Binance server status (e.g., maintenance)
    account_status = 0  # Account status for trading (e.g., sufficient funds)
    trade_state_status = 0  # Errors in trading logic

    # Latest analysis data
    df = None

    # Trade simulator variables
    transaction = None
    status = None  # Trade status (e.g., BOUGHT, SOLD)
    order = None  # Latest or current order
    order_time = None  # Order submission time

    # Account assets available for trading
    base_quantity = "0.04108219"  # BTC owned
    quote_quantity = "1000.0"  # USDT available for trade

    # Trader state and Binance account information
    system_status = {"status": 0, "msg": "normal"}
    symbol_info = {}
    account_info = {}

    # Configuration parameters
    config = {
        "api_key": "",
        "api_secret": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "merge_file_name": "data.csv",
        "feature_file_name": "features.csv",
        "matrix_file_name": "matrix.csv",
        "predict_file_name": "pred.npy",
        "signal_file_name": "signals.csv",
        "signal_models_file_name": "signal_models",
        "model_folder": "MODELS",
        "time_column": "timestamp",
        "data_folder": "/var/www",
        "symbol": "EURUSD",
        "freq": "5min",
        "data_sources": [],
        "feature_sets": [],
        "label_sets": [],
        "label_horizon": 0,
        "train_length": 0,
        "train_features": [],
        "labels": [],
        "algorithms": [],
        "features_horizon": 10,
        "signal_sets": [],
        "score_notification_model": {},
        "diagram_notification_model": {},
        "trade_model": {
            "no_trades_only_data_processing": False,
            "test_order_before_submit": False,
            "simulate_order_execution": False,
            "percentage_used_for_trade": 99,
            "limit_price_adjustment": 0.005,
        },
        "train_signal_model": {},
        "base_asset": "",
        "quote_asset": "",
        "collector": {
            "folder": "DATA",
            "flush_period": 300,
            "depth": {
                "folder": "DEPTH",
                "symbols": ["BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"],
                "limit": 100,
                "freq": "1min",
            },
            "stream": {
                "folder": "STREAM",
                "channels": ["kline_1m", "depth20"],
                "symbols": ["BTCUSDT", "ETHBTC", "ETHUSDT", "IOTAUSDT", "IOTABTC", "IOTAETH"],
            }
        },
    }


def data_provider_problems_exist() -> bool:
    """Check if there are any data provider issues."""
    return any([App.error_status, App.server_status])


def problems_exist() -> bool:
    """Check if there are any issues with system, server, account, or trading state."""
    return any([App.error_status, App.server_status, App.account_status, App.trade_state_status])


def load_config(config_file: str) -> None:
    """Load configuration from the provided file."""
    if config_file:
        config_file_path = PACKAGE_ROOT / config_file
        with open(config_file_path, encoding='utf-8') as json_file:
            conf_str = json_file.read()
            conf_str = re.sub(r"//.*$", "", conf_str, flags=re.M)
            conf_json = json.loads(conf_str)
            App.config.update(conf_json)


def load_last_transaction() -> dict:
    """Load the last transaction from file or create a default entry if no file exists."""
    transaction_file = Path("transactions.txt")
    t_dict = dict(timestamp=str(datetime.now()), price=0.0, profit=0.0, status="")
    if transaction_file.is_file():
        with open(transaction_file, "r") as f:
            line = ""
            for line in f:
                pass
        if line:
            t_dict = dict(zip("timestamp,price,profit,status".split(","), line.strip().split(",")))
            t_dict["price"] = float(t_dict["price"])
            t_dict["profit"] = float(t_dict["profit"])
    return t_dict


def load_all_transactions() -> pd.DataFrame:
    """Load all transactions from file into a DataFrame."""
    transaction_file = Path("transactions.txt")
    df = pd.read_csv(transaction_file, names="timestamp,price,profit,status".split(","), header=None)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    return df


if __name__ == "__main__":
    pass
