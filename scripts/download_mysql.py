import pandas as pd
import os
import json
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import click
import mysql.connector
from mysql.connector import Error


@click.command()
@click.option('--config_file', '-c', type=click.Path(exists=True), default='configs/config.json',
              help='Configuration file name')
@click.option('--output_format', '-f', type=click.Choice(['csv', 'json']), default='csv', help='Output file format')
def main(config_file, output_format):
    """
    Script for retrieving data from MySQL ForexFactory database and saving it locally.
    """
    # Load configuration
    with open(config_file, 'r') as file:
        config = json.load(file)

    mysql_config = config.get("mysql", {})
    base_output_folder = Path(config.get("data_folder", "data"))
    symbols = config.get("symbols", [])
    inception_date_str = config.get("inception_date", "1970-01-01")

    # Convert inception_date to timestamp in milliseconds
    inception_date = datetime.strptime(inception_date_str, "%Y-%m-%d")
    inception_timestamp = int(inception_date.timestamp() * 1000)

    # Create base output directory if it doesn't exist
    base_output_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Establish MySQL connection
        connection = mysql.connector.connect(
            host=mysql_config.get("host", "localhost"),
            port=mysql_config.get("port", 3306),
            user=mysql_config["user"],
            password=mysql_config["password"],
            database=mysql_config["database"]
        )

        if connection.is_connected():
            print(f"Connected to MySQL database '{mysql_config['database']}' successfully.")

            cursor = connection.cursor(dictionary=True)

            for symbol_data in tqdm(symbols, desc="Processing Symbols"):
                symbol = symbol_data.get('symbol')
                data_sources = symbol_data.get('data_sources', [])

                if not symbol or not data_sources:
                    print(f"Skipping invalid symbol configuration: {symbol_data}")
                    continue

                # Create folder for the symbol
                symbol_folder = base_output_folder / symbol_data.get('folder', symbol)
                symbol_folder.mkdir(parents=True, exist_ok=True)

                for data_source in data_sources:
                    file_type = data_source.get('file')
                    column_prefix = data_source.get('column_prefix', '')
                    timeframe = data_source.get('timeframe', '')  # Applicable for klines

                    if not file_type:
                        print(f"Skipping data source with missing 'file' field: {data_source}")
                        continue

                    # Determine table based on file_type
                    if file_type.lower() == 'ticks':
                        table = "MarketBDSwiss"
                        columns = ["timestamp", "ask", "bid"]
                        query = f"""
                            SELECT M.timestamp, M.ask, M.bid
                            FROM {table} AS M
                            JOIN Symbols AS S ON M.symbol_id = S.symbol_id
                            WHERE S.symbol_name = %s AND M.timestamp >= %s
                            ORDER BY M.timestamp ASC;
                        """
                        params = (symbol, inception_timestamp)

                        cursor.execute(query, params)
                        rows = cursor.fetchall()

                        if rows:
                            df = pd.DataFrame(rows)
                            # Convert timestamp from milliseconds to datetime
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            # Add column prefix
                            df.rename(columns=lambda x: f"{column_prefix}{x}" if x in columns else x, inplace=True)
                        else:
                            df = pd.DataFrame()
                            print(f"No ticks data found for symbol '{symbol}'.")

                    elif file_type.lower() == 'klines':
                        # Since 'klines' data is not defined in the schema, we'll aggregate 'ticks' into klines
                        table = "MarketBDSwiss"
                        columns = ["timestamp", "ask", "bid"]
                        query = f"""
                            SELECT M.timestamp, M.ask, M.bid
                            FROM {table} AS M
                            JOIN Symbols AS S ON M.symbol_id = S.symbol_id
                            WHERE S.symbol_name = %s AND M.timestamp >= %s
                            ORDER BY M.timestamp ASC;
                        """
                        params = (symbol, inception_timestamp)

                        cursor.execute(query, params)
                        rows = cursor.fetchall()

                        if rows:
                            df = pd.DataFrame(rows)
                            # Convert timestamp from milliseconds to datetime
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            # Set timestamp as index for resampling
                            df.set_index('timestamp', inplace=True)
                            # Aggregate to specified timeframe (default 5min)
                            timeframe = timeframe if timeframe else '5min'
                            klines = df.resample(timeframe).agg({
                                'ask': 'mean',
                                'bid': 'mean'
                            }).dropna().reset_index()
                            # Rename columns with prefix
                            klines.rename(
                                columns=lambda x: f"{column_prefix}{x}" if x in ['timestamp', 'ask', 'bid'] else x,
                                inplace=True)
                            df = klines
                        else:
                            df = pd.DataFrame()
                            print(f"No klines data found for symbol '{symbol}'.")

                    else:
                        print(f"Unsupported file type '{file_type}' for symbol '{symbol}'. Skipping.")
                        continue

                    if not df.empty:
                        # Define file path
                        file_name = f"{file_type}.{output_format}"
                        file_path = symbol_folder / file_name

                        # Save to desired format
                        if output_format == 'csv':
                            df.to_csv(file_path, index=False)
                        elif output_format == 'json':
                            df.to_json(file_path, orient='records', date_format='iso')

                        print(f"Saved '{file_type}' data for symbol '{symbol}' to '{file_path}'.")
                    else:
                        print(f"Skipping saving for '{file_type}' of symbol '{symbol}' as it contains no data.")

        else:
            print("Failed to connect to the database.")

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed.")


if __name__ == '__main__':
    main()
