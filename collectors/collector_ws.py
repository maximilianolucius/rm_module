# Binance stuff


import time

from binance.exceptions import *
from binance.client import Client
from binance.websockets import BinanceSocketManager

from service.App import *
from service.analyzer import *

import logging
log = logging.getLogger('collector_ws')


def process_message(msg):
    """Process incoming message from WebSocket stream."""
    if msg is None:
        print(f"Empty message received")
        return
    if not isinstance(msg, dict):
        print(f"Message received is not dict")
        return
    if len(msg.keys()) != 2:
        print(f"Message received has unexpected length. Message: {msg}")
        return

    error = msg.get('e')
    if error is not None:
        error_message = msg.get('m')
        print(f"Connection error: {error_message}")
        return

    stream = msg.get('stream')
    if stream is None:
        print(f"Empty stream received. Message: {msg}")
        # TODO: Check what happens and maybe reconnect
        return
    stream_symbol, stream_channel = tuple(stream.split("@"))

    event = msg.get('data')
    if event is None:
        print(f"Empty event received. Message {msg}")
        # TODO: Check what happens and maybe reconnect
        return

    event_channel = event.get('e')
    if event_channel == 'error':
        # close and restart the socket
        return
    if event_channel is None:
        event["e"] = stream_channel

    event_symbol = event.get('s')
    if event_symbol is None:
        event["s"] = stream_symbol.upper()

    event_ts = event.get('E')
    if event_ts is None:
        event["E"] = int(datetime.utcnow().timestamp() * 1000)

    # Submit a task to our main event queue
    App.analyzer.queue.put(event)


def start_collector_ws():
    """Initialize WebSocket data collection and start event listener."""
    print(f"Start collecting data using WebSocket streams.")

    # Initialize data state, connections, and listeners
    App.analyzer = Analyzer(None)
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    # Define and set up WebSocket channels and symbols
    channels = App.config["collector"]["stream"]["channels"]
    print(f"Channels: {channels}")

    symbols = App.config["collector"]["stream"]["symbols"]
    print(f"Symbols: {symbols}")

    streams = []
    for c in channels:
        for s in symbols:
            stream = s.lower() + "@" + c.lower()
            streams.append(stream)
    print(f"Streams: {streams}")

    App.bm = BinanceSocketManager(App.client, user_timeout=BinanceSocketManager.DEFAULT_USER_TIMEOUT)
    App.conn_key = App.bm.start_multiplex_socket(streams, process_message)
    App.bm.start()
    print(f"Subscribed to the streams.")

    # Periodically call database storage
    saving_period = App.config["collector"]["flush_period"]
    try:
        while True:
            time.sleep(saving_period)
            event_count = App.analyzer.queue.qsize()
            if event_count > 0:
                print(f"Storing {event_count} events.")
                App.analyzer.store_queue()
            else:
                # Attempt to reconnect
                print(f"No incoming messages. Trying to reconnect.")
                reconnect_pause = 30
                time.sleep(reconnect_pause)
                if App.bm is not None:
                    try:
                        App.bm.close()
                    except:
                        pass

                try:
                    App.bm = BinanceSocketManager(App.client, user_timeout=BinanceSocketManager.DEFAULT_USER_TIMEOUT)
                    App.conn_key = App.bm.start_multiplex_socket(streams, process_message)
                    App.bm.start()
                except:
                    print(f"Exception while reconnecting. Will try next time.")
    except KeyboardInterrupt:
        pass

    if App.bm is not None:
        App.bm.close()

    print(f"End collecting data using WebSocket streams.")
    return 0


if __name__ == "__main__":
    start_collector_ws()
