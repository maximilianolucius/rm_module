import pandas as pd
import math
import os.path
import json
import time
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook
import click

import asyncio
from binance.client import Client
from binance.streams import BinanceSocketManager
from binance.enums import *

from common.utils import klines_to_df, binance_freq_from_pandas
from service.App import *

"""
Script for retrieving Binance data, including historical klines and exchange info.
"""

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Retrieves historical klines from Binance.
    """
    load_config(config_file)
    time_column = App.config["time_column"]
    data_path = Path(App.config["data_folder"])
    now = datetime.now()

    # Set frequency for both pandas and Binance formats
    freq = App.config["freq"]
    print(f"Pandas frequency: {freq}")
    freq = binance_freq_from_pandas(freq)
    print(f"Binance frequency: {freq}")

    save = True
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    futures = False
    if futures:
        App.client.API_URL = "https://fapi.binance.com/fapi"
        App.client.PRIVATE_API_VERSION = "v1"
        App.client.PUBLIC_API_VERSION = "v1"

    data_sources = App.config["data_sources"]
    for ds in data_sources:
        quote = ds.get("folder")
        if not quote:
            print("ERROR. Folder is not specified.")
            continue

        print(f"Start downloading '{quote}' ...")
        file_path = data_path / quote
        file_path.mkdir(parents=True, exist_ok=True)

        file_name = (file_path / ("futures" if futures else "klines")).with_suffix(".csv")

        # Retrieve the latest available timestamp
        latest_klines = App.client.get_klines(symbol=quote, interval=freq, limit=5)
        latest_ts = pd.to_datetime(latest_klines[-1][0], unit='ms')

        if file_name.is_file():
            # Load existing data to append new data
            df = pd.read_csv(file_name)
            df[time_column] = pd.to_datetime(df[time_column], format='ISO8601')
            oldest_point = df["timestamp"].iloc[-5]  # Append with a slight overlap
            print(f"Appending data for {quote} from {str(latest_ts)} to existing file {file_name}")
        else:
            # Create a new file for fresh data download
            df = pd.DataFrame()
            oldest_point = datetime(2017, 1, 1)
            print(f"Creating new file for {quote} data from {freq}.")

        # Download data from Binance
        klines = App.client.get_historical_klines(
            symbol=quote,
            interval=freq,
            start_str=oldest_point.isoformat()
        )

        df = klines_to_df(klines, df)
        df = df.iloc[:-1]  # Remove last row if incomplete

        if save:
            df.to_csv(file_name)

        print(f"Downloaded '{quote}' data stored in '{file_name}'")

    elapsed = datetime.now() - now
    print(f"Completed data download in {str(elapsed).split('.')[0]}")
    return df

def get_exchange_info():
    """
    Retrieves and saves exchange information from Binance.
    """
    exchange_info = App.client.get_exchange_info()
    with open("exchange_info.json", "w") as file:
        json.dump(exchange_info, file, indent=4)

def get_account_info():
    """
    Fetches account details, including orders, trades, and account status.
    """
    orders = App.client.get_all_orders(symbol='BTCUSDT')
    trades = App.client.get_my_trades(symbol='BTCUSDT')
    info = App.client.get_account()
    status = App.client.get_account_status()
    details = App.client.get_asset_details()

def get_market_info():
    """
    Retrieves order book depth for BTCUSDT.
    """
    depth = App.client.get_order_book(symbol='BTCUSDT')

def minutes_of_new_data(symbol, freq, data):
    """
    Calculates the time range for new data to download based on the last entry.
    """
    if len(data) > 0:
        old = data["timestamp"].iloc[-1]
    else:
        old = datetime.strptime('1 Jan 2017', '%d %b %Y')

    # Retrieve latest kline timestamp
    new_info = App.client.get_klines(symbol=symbol, interval=freq)
    new = pd.to_datetime(new_info[-1][0], unit='ms')
    return old, new

async def get_futures_klines_all(symbol, freq, save=False):
    """
    Async function to fetch futures klines and save to CSV if specified.
    """
    filename = "futures.csv"
    if os.path.isfile(filename):
        data_df = pd.read_csv(filename)
    else:
        data_df = pd.DataFrame()

    import hmac
    import hashlib
    import urllib.parse

    headers = {"X-MBX-APIKEY": binance_api_key}
    params = {}
    query = urllib.parse.urlencode(params)
    signature = hmac.new(
        binance_api_secret.encode("utf8"),
        query.encode("utf8"),
        digestmod=hashlib.sha256
    ).hexdigest()
    params["signature"] = signature

    # Test connection
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, params=params) as response:
            result = await response.json()

    # Retrieve klines for specified interval
    start = 1483228800000  # Start time in ms for 2017
    end = 1569888000000    # End time in ms for 2019-10-07

    params = {"symbol": symbol, "interval": freq, "startTime": start, "endTime": end}
    query = urllib.parse.urlencode(params)
    signature = hmac.new(
        binance_api_secret.encode("utf8"),
        query.encode("utf8"),
        digestmod=hashlib.sha256
    ).hexdigest()
    params["signature"] = signature

    url = "https://fapi.binance.com/fapi/v1/klines"
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, params=params) as response:
            klines = await response.json()

    data_df = klines_to_df(klines, None)
    if save:
        data_df.to_csv(filename)

def check_market_stream():
    """
    Starts multiple market streams (depth, kline, etc.) from Binance.
    """
    bm = BinanceSocketManager(App.client)
    conn_key = bm.start_trade_socket('BTCUSDT', message_fn)
    conn_key = bm.start_kline_socket('BTCUSDT', message_fn, interval=KLINE_INTERVAL_30MINUTE)
    bm.start()
    time.sleep(10)
    bm.stop_socket(conn_key)
    bm.close()
    from twisted.internet import reactor
    reactor.stop()

def message_fn(msg):
    """
    Prints incoming message from the WebSocket.
    """
    print(msg)

def check_market_stream_multiplex():
    """
    Starts multiplexed WebSocket streams for multiple symbols.
    """
    bm = BinanceSocketManager(App.client)
    conn_key = bm.start_multiplex_socket(['bnbbtc@aggTrade', 'neobtc@ticker'], multiples_fn)

def multiples_fn(msg):
    """
    Processes multiplexed WebSocket messages.
    """
    print("stream: {} data: {}".format(msg['stream'], msg['data']))

def check_user_stream():
    """
    Opens user-specific WebSocket streams (account, order, and trade events).
    """
    bm = BinanceSocketManager(App.client)
    bm.start_user_socket(user_message_fn)

def user_message_fn(msg):
    """
    Processes messages from user WebSocket stream.
    """
    print(msg)

if __name__ == '__main__':
    main()
