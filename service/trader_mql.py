import os
import sys
import json
import asyncio
import logging
from decimal import Decimal
from datetime import datetime

from mql_client import mql_client  # Ensure mql_client.py is in the same directory or properly referenced
from common.utils import round_str, round_down_str, to_decimal  # Adjust import paths as necessary
from service.analyzer import Analyzer  # Placeholder for your analyzer module
from service.notifier_trades import get_signal  # Placeholder for your signal generation module

# Initialize Logging
def setup_logging(log_file, log_level):
    """
    Sets up logging for the trader.

    Args:
        log_file (str): Path to the log file.
        log_level (str): Logging level as a string.

    Returns:
        Logger: Configured logger instance.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO

    logging.basicConfig(
        filename=log_file,
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('trader')

# Event Handler Class
class TradeEventHandler:
    """
    Handles events triggered by the mql_client.
    """

    def on_order_event(self):
        """
        Handle order-related events.
        """
        logger.info("Order event detected.")
        # Implement your order event handling logic here

    def on_message(self, message):
        """
        Handle incoming messages from MetaTrader.

        Args:
            message (dict): The message received.
        """
        logger.info(f"Message received: {message}")
        # Implement your message handling logic here

    def on_tick(self, symbol, bid, ask):
        """
        Handle tick data updates.

        Args:
            symbol (str): Trading symbol.
            bid (float): Bid price.
            ask (float): Ask price.
        """
        logger.debug(f"Tick update for {symbol}: Bid={bid}, Ask={ask}")
        # Implement your tick handling logic here

    def on_bar_data(self, symbol, time_frame, time, open_price, high, low, close, tick_volume):
        """
        Handle bar data updates.

        Args:
            symbol (str): Trading symbol.
            time_frame (str): Timeframe of the bar (e.g., 'M1', 'H1').
            time (str): Time of the bar.
            open_price (float): Open price.
            high (float): High price.
            low (float): Low price.
            close (float): Close price.
            tick_volume (int): Tick volume.
        """
        logger.debug(f"Bar data for {symbol} [{time_frame}]: {time}, O={open_price}, H={high}, L={low}, C={close}, V={tick_volume}")
        # Implement your bar data handling logic here

    def on_historic_data(self, symbol, time_frame, data):
        """
        Handle historic data updates.

        Args:
            symbol (str): Trading symbol.
            time_frame (str): Timeframe of the data.
            data (dict): Historic data.
        """
        logger.info(f"Historic data received for {symbol} [{time_frame}].")
        # Implement your historic data handling logic here

    def on_historic_trades(self):
        """
        Handle historic trades updates.
        """
        logger.info("Historic trades data received.")
        # Implement your historic trades handling logic here

# Main Trading Bot Class
class TradeBot:
    def __init__(self, config):
        """
        Initializes the trading bot with configuration parameters.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.symbol = config["symbol"]
        self.trade_model = config.get("trade_model", {})
        self.sleep_delay = config.get("trade_settings", {}).get("sleep_delay", 1)
        self.max_retry_command_seconds = config.get("trade_settings", {}).get("max_retry_command_seconds", 10)

        # Initialize Analyzer (Placeholder - implement as needed)
        self.analyzer = Analyzer(config)

        # Initialize mql_client with event handler
        self.event_handler = TradeEventHandler()
        self.mql = mql_client(
            event_handler=self.event_handler,
            metatrader_dir_path=config["metatrader_dir_path"],
            sleep_delay=self.sleep_delay,
            max_retry_command_seconds=self.max_retry_command_seconds,
            load_orders_from_file=True,
            verbose=True
        )

        # Initialize account balances
        self.base_quantity = Decimal('0')
        self.quote_quantity = Decimal('0')
        self.order = {}
        self.status = "NONE"

        # Subscribe to symbols and bar data
        self.mql.subscribe_symbols([self.symbol])
        self.mql.subscribe_symbols_bar_data([[self.symbol, 'M5']])  # Example: 5-minute timeframe

    async def main_trader_task(self):
        """
        The primary task for the trading bot, executed periodically to handle trades.
        """
        while True:
            try:
                signal = get_signal()  # Implement your signal generation logic
                signal_side = signal.get("side")
                close_price = signal.get("close_price")
                close_time = signal.get("close_time")

                logger.info(f"===> Start trade task. Signal: {signal_side} at {close_price}.")

                # Sync trade status and check orders
                status = self.status
                if status in ["BUYING", "SELLING"]:
                    order_status = await self.update_order_status()
                    if not self.order or not order_status:
                        await self.update_trade_status()
                        logger.error("Order issue. Reset needed.")
                        continue

                    if order_status == "FILLED":
                        if status == "BUYING":
                            self.status = "BOUGHT"
                        elif status == "SELLING":
                            self.status = "SOLD"
                        logger.info(f"New trade mode: {self.status}")
                    elif order_status in ["REJECTED", "EXPIRED", "CANCELED"]:
                        self.status = "SOLD" if status == "BUYING" else "BOUGHT"
                        logger.info(f"New trade mode: {self.status}")
                else:
                    logger.error(f"Unexpected status value: {status}.")

                # Prepare, cancel existing orders if needed
                if status in ["BUYING", "SELLING"]:
                    if await self.cancel_order():
                        self.status = "SOLD" if status == "BUYING" else "BOUGHT"

                # Handle new signals
                if signal_side in ["BUY", "SELL"]:
                    await self.update_account_balance()
                    if self.status == "SOLD" and signal_side == "BUY":
                        await self.new_market_order(side="BUY")
                        if not self.trade_model.get("no_trades_only_data_processing"):
                            self.status = "BUYING"
                    elif self.status == "BOUGHT" and signal_side == "SELL":
                        await self.new_market_order(side="SELL")
                        if not self.trade_model.get("no_trades_only_data_processing"):
                            self.status = "SELLING"

                logger.info(f"<=== End trade task.")

            except Exception as e:
                logger.error(f"Error in main_trader_task: {e}")
                logger.debug("Traceback:", exc_info=True)

            await asyncio.sleep(self.sleep_delay)

    async def update_trade_status(self):
        """Update account and trade status based on open orders."""
        try:
            open_orders = self.mql.open_orders
            if not open_orders:
                await self.update_account_balance()
                last_kline = self.analyzer.get_last_kline(self.symbol)
                last_close_price = to_decimal(last_kline['close'])

                btc_assets_in_usd = self.base_quantity * last_close_price
                usd_assets = self.quote_quantity

                self.status = "SOLD" if usd_assets >= btc_assets_in_usd else "BOUGHT"
                logger.info(f"Updated trade status to: {self.status}")
            else:
                first_order = next(iter(open_orders.values()))
                self.status = "SELLING" if first_order.get("side") == "SELL" else "BUYING"
                logger.info(f"Updated trade status to: {self.status}")
        except Exception as e:
            logger.error(f"Error updating trade status: {e}")

    async def update_order_status(self):
        """Update the current order's status and return it."""
        order_id = self.order.get("orderId", 0) if self.order else 0
        if not order_id:
            logger.error("Invalid order ID.")
            return None

        try:
            # Fetch order status from mql_client's open_orders
            order_info = self.mql.open_orders.get(str(order_id), {})
            order_status = order_info.get("status", None)
            if order_status:
                logger.debug(f"Order ID {order_id} status: {order_status}")
                return order_status
            else:
                logger.warning(f"Order ID {order_id} not found in open orders.")
                return None
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None

    async def update_account_balance(self):
        """Fetch and update the current account balance."""
        try:
            # Fetch account balance from mql_client's account_info
            balance = self.mql.account_info
            self.base_quantity = Decimal(balance.get("base_asset_free", "0.000000"))
            self.quote_quantity = Decimal(balance.get("quote_asset_free", "0.000000"))
            logger.debug(f"Updated balances: {self.base_quantity} {self.config['base_asset']}, {self.quote_quantity} {self.config['quote_asset']}")
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")

    async def cancel_order(self):
        """Cancel the active order."""
        order_id = self.order.get("orderId", 0) if self.order else 0
        if not order_id:
            logger.error("No active order to cancel.")
            return False

        try:
            # Send cancel order command via mql_client
            self.mql.close_order(ticket=order_id, lots=0)
            logger.info(f"Sent cancel command for order ID {order_id}.")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def new_market_order(self, side):
        """Create a new market order."""
        try:
            last_kline = self.analyzer.get_last_kline(self.symbol)
            last_close_price = to_decimal(last_kline['close'])
            if not last_close_price:
                logger.error("Unable to retrieve close price.")
                return None

            # Market orders don't require price adjustments
            # Calculate quantity based on side
            if side == "BUY":
                quantity = self.quote_quantity / last_close_price
            elif side == "SELL":
                quantity = self.base_quantity
            else:
                logger.error(f"Invalid order side: {side}")
                return None

            quantity_str = round_down_str(quantity, 6)

            # Prepare order parameters
            order_spec = {
                "symbol": self.symbol,
                "order_type": "market",
                "side": side.lower(),
                "lots": float(quantity_str),
                "price": 0,  # Not needed for market orders
                "stop_loss": 0,  # Implement as needed
                "take_profit": 0,  # Implement as needed
                "magic": 0,  # Implement as needed
                "comment": "",
                "expiration": 0
            }

            # Send order via mql_client
            self.mql.open_order(
                symbol=self.symbol,
                order_type="market",
                side=side.lower(),
                lots=float(quantity_str),
                price=order_spec["price"],  # Not used in market orders
                stop_loss=order_spec["stop_loss"],
                take_profit=order_spec["take_profit"],
                magic=order_spec["magic"],
                comment=order_spec["comment"],
                expiration=order_spec["expiration"]
            )
            logger.info(f"Sent new {side} market order: {order_spec}")
            self.order = order_spec  # Update current order
            return order_spec

        except Exception as e:
            logger.error(f"Error creating new market order: {e}")
            return None

# Utility Functions (Placeholder - Implement as needed)
# Ensure these are defined in your common.utils module or adjust accordingly
def round_str(price, decimals):
    return f"{price:.{decimals}f}"

def round_down_str(quantity, decimals):
    format_str = f"{{0:.{decimals}f}}"
    return format_str.format(quantity)

def to_decimal(value):
    try:
        return Decimal(str(value))
    except:
        return Decimal('0')

# Initialize Logger
logger = setup_logging(
    log_file="trade_mq4.log",
    log_level="DEBUG"  # This will be updated based on config
)

# Load Configuration
def load_config(config_file_path):
    """
    Loads the configuration from a JSON file.

    Args:
        config_file_path (str): Path to the configuration JSON file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file {config_file_path} does not exist.")
        sys.exit(1)

    with open(config_file_path, 'r') as f:
        config = json.load(f)
    return config

# Main Function
def main():
    """
    Entry point for the trading bot.
    """
    if len(sys.argv) < 2:
        print("Usage: python trade_mq4.py <path_to_config.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    # Update logger based on config
    logger.handlers = []  # Remove existing handlers
    global logger
    logger = setup_logging(
        log_file=config.get("logging", {}).get("log_file", "trade_mq4.log"),
        log_level=config.get("logging", {}).get("log_level", "DEBUG")
    )

    logger.info("Starting TradeBot with configuration.")

    # Initialize TradeBot
    trade_bot = TradeBot(config)

    # Start the main trader task
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(trade_bot.main_trader_task())
    except KeyboardInterrupt:
        logger.info("TradeBot stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        loop.close()

if __name__ == "__main__":
    main()
