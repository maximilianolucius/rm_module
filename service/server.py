import os
import sys
from datetime import timedelta, datetime
import asyncio
import logging
import click
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from decimal import *

from service.App import *
from common.utils import *
from service.analyzer import *
from service.notifier_trades import *
from service.trader_mql import TradeBot

log = logging.getLogger('server')
trader_bot = None

async def main_task():
    # """
    # Scheduled main task that collects data, analyzes it, and trading actions.
    # """
    # res = await main_collector_task()
    # if res:
    #     return res

    try:
        analyze_task = await App.loop.run_in_executor(None, App.analyzer.analyze)
    except Exception as e:
        log.error(f"Error while analyzing data: {e}")
        return

    trade_model = App.config.get("trade_model", {})
    if trade_model.get("trader_mql"):
        App.loop.create_task(trader_bot.main_trader_task())

    return

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def start_server(config_file):
    """
    Initializes and starts the trading server with data collection, analysis, notifications, and trading tasks.
    """
    load_config(config_file)
    symbol = App.config["symbol"]
    freq = App.config["freq"]
    trader_bot = TradeBot(config_file)

    print(f"Initializing server. Trade pair: {symbol}.")
    App.client = trader_bot
    App.analyzer = Analyzer(App.config)
    App.loop = asyncio.get_event_loop()

    try:
        App.loop.run_until_complete(trader_bot.main_trader_task())
    except Exception as e:
        log.error(f"Health check error: {e}")

    if data_provider_problems_exist():
        log.error("Data provider issues detected during health check.")
        return

    try:
        # App.loop.run_until_complete(sync_data_collector_task())
        App.analyzer.analyze(ignore_last_rows=True)
    except Exception as e:
        log.error(f"Initial data collection issue: {e}")

    if data_provider_problems_exist():
        log.error("Issues encountered during initial data collection.")
        return

    if App.config.get("trade_model", {}).get("trader_binance"):
        try:
            App.loop.run_until_complete(trader_bot.update_trade_status())
        except Exception as e:
            log.error(f"Trade status sync error: {e}")

        if data_provider_problems_exist():
            log.error("Issues encountered during trade status sync.")
            return

        print(f"Balance: {App.config['base_asset']} = {str(App.base_quantity)}")
        print(f"Balance: {App.config['quote_asset']} = {str(App.quote_quantity)}")

    App.sched = AsyncIOScheduler()
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    trigger = freq_to_CronTrigger(freq)
    App.sched.add_job(main_task, trigger=trigger, id='main_task')
    App.sched.start()

    try:
        App.loop.run_forever()
    except KeyboardInterrupt:
        log.info("Server interrupted by user.")
    finally:
        App.loop.close()
        App.sched.shutdown()
        log.info("Server shutdown.")

if __name__ == "__main__":
    start_server()
