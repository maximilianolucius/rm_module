# Binance stuff

import asyncio

from apscheduler.schedulers.background import BackgroundScheduler

from binance.exceptions import *
from binance.client import Client

from service.analyzer import *

import logging
log = logging.getLogger('collector_depth')


async def main_collector_depth_task():
    """Execute depth collection task for each depth collection cycle, e.g., every 5 seconds."""
    log.info(f"===> Start depth collection task.")
    start_time = datetime.utcnow()

    # Get parameters for data collection from the config
    symbols = App.config["collector"]["depth"]["symbols"]
    limit = App.config["collector"]["depth"]["limit"]
    freq = App.config["collector"]["depth"]["freq"]

    # Submit tasks for requesting data and process results
    tasks = [asyncio.create_task(request_depth(sym, freq, limit)) for sym in symbols]

    results = []
    timeout = 3  # Timeout in seconds

    # Process responses in the order of arrival
    for fut in asyncio.as_completed(tasks, timeout=timeout):
        try:
            res = await fut
            results.append(res)
            try:
                # Add response to the database
                added_count = App.analyzer.store_depth([res], freq)
            except Exception as e:
                log.error(f"Error storing order book result in the database.")
        except TimeoutError as te:
            log.warning(f"Timeout {timeout} seconds when requesting order book data.")
        except Exception as e:
            log.warning(f"Exception when requesting order book data.")

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    log.info(f"<=== End depth collection task. {len(results)} responses stored. {duration:.2f} seconds processing time.")


async def request_depth(symbol, freq, limit):
    """Request order book data from the Binance API for a given symbol."""
    requestTime = now_timestamp()

    depth = {}
    try:
        depth = App.client.get_order_book(symbol=symbol, limit=limit)  # Request with weight=1 for <100 limit
    except BinanceRequestException as bre:
        try:
            depth['code'] = bre.code
            depth['msg'] = bre.msg
        except:
            pass
    except BinanceAPIException as bae:
        try:
            depth['code'] = bae.code
            depth['msg'] = bae.message
        except:
            pass

    responseTime = now_timestamp()

    # Post-process data
    depth['timestamp'] = pandas_get_interval(freq=freq, timestamp=requestTime)[0]
    depth['requestTime'] = requestTime
    depth['responseTime'] = responseTime
    depth['symbol'] = symbol

    return depth


def start_collector_depth():
    """Initialize and start the depth data collector scheduler and event loop."""
    App.analyzer = Analyzer(None)
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])

    # Register scheduler and jobs
    App.sched = BackgroundScheduler(daemon=False)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

    # Schedule job based on frequency in config
    freq = App.config["collector"]["depth"]["freq"]

    if freq == "5s":
        App.sched.add_job(
            lambda: asyncio.run_coroutine_threadsafe(main_collector_depth_task(), App.loop),
            trigger='cron',
            second='*/5',
            id='sync_depth_collector_task'
        )
    elif freq == "1m":
        App.sched.add_job(
            lambda: asyncio.run_coroutine_threadsafe(main_collector_depth_task(), App.loop),
            trigger='cron',
            minute='*',
            id='sync_depth_collector_task'
        )
    else:
        log.error(f"Unknown frequency in app config: {freq}. Exit.")
        return

    App.sched.start()  # Start the scheduler thread

    # Start asyncio event loop
    App.loop = asyncio.get_event_loop()
    try:
        App.loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("===> Closing Loop")
        App.loop.close()
        App.sched.shutdown()

    return 0


if __name__ == "__main__":
    App.analyzer = Analyzer(None)
    App.client = Client(api_key=App.config["api_key"], api_secret=App.config["api_secret"])
    App.loop = asyncio.get_event_loop()
    try:
        App.loop.run_until_complete(main_collector_depth_task())
    except KeyboardInterrupt:
        pass
    finally:
        print("===> Closing Loop")
        App.loop.close()
        App.sched.shutdown()
