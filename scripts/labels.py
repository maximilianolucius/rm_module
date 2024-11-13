from pathlib import Path
import pandas as pd
import click
from service.App import *
from scripts.features import generate_feature_set

class P:
    in_nrows = 100_000_000
    tail_rows = 0

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    """
    Load a data file with close prices, compute top-bottom labels, add labels to the data,
    and save to the specified output file.
    """
    load_config(config_file)
    time_column = App.config["time_column"]
    now = datetime.now()

    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    file_path = data_path / App.config.get("feature_file_name")

    if not file_path.is_file():
        print(f"Data file does not exist: {file_path}")
        return

    print(f"Loading data from {file_path}...")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=P.in_nrows)
    else:
        print(f"ERROR: Unsupported file extension '{file_path.suffix}'. Only 'csv' and 'parquet' are supported.")
        return
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    df = df.iloc[-P.tail_rows:]
    df = df.reset_index(drop=True)

    print(f"Data range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    label_sets = App.config.get("label_sets", [])
    if not label_sets:
        print(f"ERROR: No label sets defined.")
        return

    all_features = []
    for i, fs in enumerate(label_sets):
        fs_now = datetime.now()
        print(f"Processing label set {i + 1}/{len(label_sets)}: {fs.get('generator')}...")
        df, new_features = generate_feature_set(df, fs, last_rows=0)
        all_features.extend(new_features)
        fs_elapsed = datetime.now() - fs_now
        print(f"Generated {len(new_features)} labels in {str(fs_elapsed).split('.')[0]}.")

    print(f"Generated labels. Checking for NULL values:")
    print(df[all_features].isnull().sum().sort_values(ascending=False))

    out_file_name = App.config.get("matrix_file_name")
    out_path = (data_path / out_file_name).resolve()
    print(f"Saving to {out_path} with {len(df)} records and {len(df.columns)} columns...")
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        print(f"ERROR: Unsupported file extension '{out_path.suffix}'. Only 'csv' and 'parquet' are supported.")
        return
    print(f"Data saved to {out_path}.")

    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write(", ".join([f'"{f}"' for f in all_features]) + "\n\n")
    print(f"Label names saved to {out_path.with_suffix('.txt')}")

    elapsed = datetime.now() - now
    print(f"Completed {len(all_features)} labels in {str(elapsed).split('.')[0]}.")

if __name__ == '__main__':
    main()
