from pathlib import Path

import numpy as np

from common.depth_processing import *
from common.gen_features import *

symbol = "BTCUSDT"
in_path_name = r"C:\DATA2\BITCOIN\COLLECTED\DEPTH\batch6-partial-till-0307"


def get_symbol_files(symbol):
    """
    Get a list of file names with data for this symbol and frequency.
    Files with this symbol in name are found in the directory recursively.
    """
    file_pattern = f"*{symbol}*.txt"
    paths = Path(in_path_name).rglob(file_pattern)
    return list(paths)


def find_depth_statistics():
    """
    Analyze depth data by computing statistics for price span, volume, and number of bids/asks.
    """
    paths = get_symbol_files(symbol)
    bad_lines = 0
    bid_spans, ask_spans, bid_lens, ask_lens, bid_vols, ask_vols = [], [], [], [], [], []

    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except:
                    bad_lines += 1
                    continue
                if not entry.get("bids") or not entry.get("asks"):
                    bad_lines += 1
                    continue
                bids = [float(x[0]) for x in entry.get("bids")]
                asks = [float(x[0]) for x in entry.get("asks")]
                bid_spans.append(np.max(bids) - np.min(bids))
                ask_spans.append(np.max(asks) - np.min(asks))
                bid_lens.append(len(bids))
                ask_lens.append(len(asks))
                bid_vols.append(np.sum([float(x[1]) for x in entry.get("bids")]))
                ask_vols.append(np.sum([float(x[1]) for x in entry.get("asks")]))

    print(f"Bid spans: min={np.min(bid_spans):.2f}, max={np.max(bid_spans):.2f}, mean={np.mean(bid_spans):.2f}")
    print(f"Ask spans: min={np.min(ask_spans):.2f}, max={np.max(ask_spans):.2f}, mean={np.mean(ask_spans):.2f}")
    print(f"Bid lens: min={np.min(bid_lens):.2f}, max={np.max(bid_lens):.2f}, mean={np.mean(bid_lens):.2f}")
    print(f"Ask lens: min={np.min(ask_lens):.2f}, max={np.max(ask_lens):.2f}, mean={np.mean(ask_lens):.2f}")
    print(f"Bid vols: min={np.min(bid_vols):.2f}, max={np.max(bid_vols):.2f}, mean={np.mean(bid_vols):.2f}")
    print(f"Ask vols: min={np.min(ask_vols):.2f}, max={np.max(ask_vols):.2f}, mean={np.mean(ask_vols):.2f}")
    print(f"Bad lines: {bad_lines}")


def main(args=None):
    """
    Process each file by loading, filtering, and computing features, then saving the results.
    """
    start_dt = datetime.now()
    print(f"Start processing...")

    paths = get_symbol_files(symbol)
    for path in paths:
        bad_lines = 0
        table = []

        with open(path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except:
                    bad_lines += 1
                    continue
                if not entry.get("bids") or not entry.get("asks"):
                    bad_lines += 1
                    continue
                timestamp = entry.get("timestamp")
                if timestamp % 60_000 != 0:
                    continue
                bids = [[float(x[0]), float(x[1])] for x in entry.get("bids")]
                asks = [[float(x[0]), float(x[1])] for x in entry.get("asks")]
                entry["bids"], entry["asks"] = bids, asks
                table.append(entry)

        df = depth_to_df(table)
        df = df.reset_index().rename(columns={"index": "timestamp"})
        df["timestamp"] = df["timestamp"].shift(periods=1)

        df.to_csv(path.with_suffix('.csv').name, index=False, float_format="%.4f")
        print(f"Finished processing file: {path}")
        print(f"Bad lines: {bad_lines}")

    elapsed = datetime.now() - start_dt
    print(f"Finished processing in {int(elapsed.total_seconds())} seconds.")


if __name__ == '__main__':
    main()
