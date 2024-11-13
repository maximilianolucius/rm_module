from datetime import datetime, date, timedelta
import click
import yfinance as yf
from service.App import *

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Main function to download and update financial data from Yahoo Finance using a configuration file.
    """
    load_config(config_file)
    time_column = App.config["time_column"]
    data_path = Path(App.config["data_folder"])

    now = datetime.now()
    data_sources = App.config["data_sources"]

    for ds in data_sources:
        quote = ds.get("folder")
        if not quote:
            print(f"ERROR. Folder is not specified.")
            continue

        file = ds.get("file", quote)
        if not file:
            file = quote

        print(f"Start downloading '{quote}' ...")
        file_path = data_path / quote
        file_path.mkdir(parents=True, exist_ok=True)

        file_name = (file_path / file).with_suffix(".csv")

        if file_name.is_file():
            df = pd.read_csv(file_name, parse_dates=[time_column], date_format="ISO8601")
            df[time_column] = df[time_column].dt.date
            last_date = df.iloc[-1][time_column]

            new_df = yf.download(quote, period="5d", auto_adjust=True)
            new_df = new_df.reset_index()
            new_df['Date'] = pd.to_datetime(new_df['Date'], format="ISO8601").dt.date
            new_df.rename({'Date': time_column}, axis=1, inplace=True)
            new_df.columns = new_df.columns.str.lower()

            df = pd.concat([df, new_df])
            df = df.drop_duplicates(subset=[time_column], keep="last")

        else:
            print(f"File not found. Performing full data fetch...")
            df = yf.download(quote, period="max", auto_adjust=True)
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'], format="ISO8601").dt.date
            df.rename({'Date': time_column}, axis=1, inplace=True)
            df.columns = df.columns.str.lower()

            print(f"Full fetch completed.")

        df = df.sort_values(by=time_column)
        df.to_csv(file_name, index=False)
        print(f"Data saved to '{file_name}'")

    elapsed = datetime.now() - now
    print(f"Data download completed in {str(elapsed).split('.')[0]}")

    return df

if __name__ == '__main__':
    main()
