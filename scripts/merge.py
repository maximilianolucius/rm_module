import click

from service.App import *


class P:
    """
    Parameters for handling specific data source configurations.
    """
    depth_file_names = []  # Leave empty to skip loading depth files


def load_futur_files(futur_file_path):
    """
    Load futures data from the specified file path.
    Returns a DataFrame and the start and end timestamps.
    """
    df = pd.read_csv(futur_file_path, parse_dates=['timestamp'], date_format="ISO8601")
    start, end = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    df = df.set_index("timestamp")
    print(f"Loaded futures data with {len(df)} records. Range: ({start}, {end})")
    return df, start, end


def load_kline_files(kline_file_path):
    """
    Load kline data from the specified file path.
    Returns a DataFrame and the start and end timestamps.
    """
    df = pd.read_csv(kline_file_path, parse_dates=['timestamp'], date_format="ISO8601")
    start, end = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    df = df.set_index("timestamp")
    print(f"Loaded kline data with {len(df)} records. Range: ({start}, {end})")
    return df, start, end


def load_depth_files():
    """
    Load depth data from specified file paths in P.depth_file_names.
    Returns a list of DataFrames with depth features.
    """
    dfs, start, end = [], None, None
    for depth_file_name in P.depth_file_names:
        df = pd.read_csv(depth_file_name, parse_dates=['timestamp'], date_format="ISO8601")
        start = min(start, df["timestamp"].iloc[0]) if start else df["timestamp"].iloc[0]
        end = max(end, df["timestamp"].iloc[-1]) if end else df["timestamp"].iloc[-1]
        df = df.set_index("timestamp")
        dfs.append(df)

    length = sum(len(df) for df in dfs)
    print(f"Loaded {len(P.depth_file_names)} depth files with {length} records. Range: ({start}, {end})")
    return dfs, start, end


@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Main function for merging multiple data sources into one file with a regular time raster.
    """
    load_config(config_file)
    time_column, data_sources = App.config["time_column"], App.config.get("data_sources", [])

    if not data_sources:
        print("ERROR: No data sources defined in configuration.")
        return

    now = datetime.now()
    data_path = Path(App.config["data_folder"])

    for ds in data_sources:
        quote, file = ds.get("folder"), ds.get("file", ds.get("folder"))
        file_path = (data_path / quote / file).with_suffix(".csv")

        if not file_path.is_file():
            print(f"Data file does not exist: {file_path}")
            return

        print(f"Reading data file: {file_path}")
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
        print(f"Loaded file with {len(df)} records.")
        ds["df"] = df

    df_out = merge_data_sources(data_sources)
    out_path = data_path / App.config.get("merge_file_name")

    print(f"Storing output file...")
    df_out = df_out.reset_index()
    if out_path.suffix == ".parquet":
        df_out.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df_out.to_csv(out_path, index=False)
    else:
        print(f"ERROR: Unsupported file extension '{out_path.suffix}'. Only 'csv' and 'parquet' are supported.")
        return

    range_start, range_end = df_out.index[0], df_out.index[-1]
    print(f"Stored output file {out_path} with {len(df_out)} records. Range: ({range_start}, {range_end})")
    print(f"Finished merging data in {str(datetime.now() - now).split('.')[0]}")


def merge_data_sources(data_sources: list):
    """
    Merge multiple data sources into one DataFrame with a regular time index.
    """
    time_column, freq = App.config["time_column"], App.config["freq"]

    for ds in data_sources:
        df = ds.get("df")
        if time_column in df.columns:
            df = df.set_index(time_column)
        elif df.index.name != time_column:
            print("ERROR: Timestamp column is absent.")
            return

        if ds['column_prefix']:
            df.columns = [ds['column_prefix'] + "_" + col if not col.startswith(ds['column_prefix'] + "_") else col for
                          col in df.columns]

        ds["start"], ds["end"], ds["df"] = df.first_valid_index(), df.last_valid_index(), df

    range_start, range_end = min(ds["start"] for ds in data_sources), min(ds["end"] for ds in data_sources)
    index = pd.date_range(range_start, range_end, freq=freq)
    df_out = pd.DataFrame(index=index)
    df_out.index.name = time_column

    for ds in data_sources:
        df_out = df_out.join(ds["df"])

    return df_out


if __name__ == '__main__':
    main()
