import numpy as np
import pandas as pd


def depth_to_df(depth: list):
    """
    Converts a list of market depth records into a DataFrame with computed features.
    Each record contains bids and asks, and the function processes each record to generate features.
    The resulting DataFrame has a continuous index of timestamps, matching data by shifting to align with close times.
    """
    bin_size = 1.0  # Bin size in USDT
    windows = [1, 2, 5, 10, 20]  # Number of bins for volume smoothing

    # Generate features for each record
    table = []
    for entry in depth:
        record = depth_to_features(entry, windows, bin_size)
        table.append(record)

    # Convert to DataFrame and set timestamp as index
    df = pd.DataFrame.from_dict(table)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df = df.set_index("timestamp").sort_index()

    # Define start and end times to ensure no data loss
    start_ts = depth[0].get("timestamp")
    end_ts = depth[-1].get("timestamp")

    # Define index range and join with data to fill gaps
    start = pd.to_datetime(start_ts, unit='ms')
    end = pd.to_datetime(end_ts, unit='ms')
    index = pd.date_range(start, end, freq="min")
    df_out = pd.DataFrame(index=index)
    df_out = df_out.join(df)

    return df_out


def depth_to_features(entry: list, windows: list, bin_size: float):
    """Convert one record of market depth into a dictionary of features."""
    bids = entry.get("bids")
    asks = entry.get("asks")
    timestamp = entry.get("timestamp")

    # Calculate gap and price
    gap = asks[0][0] - bids[0][0]
    gap = max(gap, 0)
    price = bids[0][0] + (gap / 2)

    # Density calculations for bids and asks
    densities = mean_volumes(depth=entry, windows=windows, bin_size=bin_size)
    record = {"timestamp": timestamp, "gap": gap, "price": price}
    record.update(densities)

    return record


def price_to_volume(side, depth, price_limit):
    """
    Calculate available volume up to a given price limit for a specified side (buy/sell).
    Returns volume if limit is reached in depth data, None otherwise.
    """
    orders = depth.get("asks", []) if side == "buy" else depth.get("bids", [])
    orders = [o for o in orders if o[0] <= price_limit] if side == "buy" else [o for o in orders if o[0] >= price_limit]

    return orders[-1][1] if orders else None


def volume_to_price(side, depth, volume_limit):
    """
    Calculate price corresponding to a specified volume limit for a given side.
    Returns price limit if reached in depth data, None otherwise.
    """
    orders = depth.get("asks", []) if side == "buy" else depth.get("bids", [])
    orders = [o for o in orders if o[1] <= volume_limit]

    return orders[-1][0] if orders else None


def depth_accumulate(depth: list, start, end):
    """
    Generate an accumulated volume curve for bid/ask volumes.
    Converts bid/ask depth lists into cumulative volumes, returning the modified depth list.
    """
    prev_value = 0.0
    for point in depth:
        point[1] += prev_value
        prev_value = point[1]

    return depth


def discretize(side: str, depth: list, bin_size: float, start: float):
    """
    Discretize depth data by binning volumes within price intervals.
    Returns volumes per price bin as a list.
    """
    price_increase = side in ["ask", "sell"]
    start = start or depth[0][0]
    bin_count = int(abs(depth[-1][0] - start) // bin_size) + 1
    end = start + bin_count * bin_size if price_increase else start - bin_count * bin_size

    bin_volumes = []
    for b in range(bin_count):
        bin_start = start + b * bin_size if price_increase else start - b * bin_size
        bin_end = bin_start + bin_size if price_increase else bin_start - bin_size

        bin_point_ids = [i for i, x in enumerate(depth) if bin_start <= x[0] < bin_end] if price_increase else [i for
                                                                                                                i, x in
                                                                                                                enumerate(
                                                                                                                    depth)
                                                                                                                if
                                                                                                                bin_end <
                                                                                                                x[
                                                                                                                    0] <= bin_start]

        bin_volume = 0.0
        prev_price = bin_start
        prev_volume = depth[bin_point_ids[0] - 1][1] if bin_point_ids and bin_point_ids[0] > 0 else 0.0

        for point_id in bin_point_ids:
            point = depth[point_id]
            price_delta = abs(point[0] - prev_price)
            bin_volume += prev_volume * (price_delta / bin_size)
            prev_price, prev_volume = point[0], point[1]

        # Contribution of the last point in the bin
        price_delta = abs(bin_end - prev_price)
        bin_volume += prev_volume * (price_delta / bin_size)
        bin_volumes.append(bin_volume)

    return bin_volumes


def mean_volumes(depth: list, windows: list, bin_size: float):
    """
    Compute mean volume density for a given depth and aggregation windows.
    Returns a dictionary with mean volume per price unit for each aggregation window.
    """
    bid_volumes = discretize(side="bid", depth=depth.get("bids"), bin_size=bin_size, start=None)
    ask_volumes = discretize(side="ask", depth=depth.get("asks"), bin_size=bin_size, start=None)

    densities = {}
    for length in windows:
        densities[f"bids_{length}"] = np.nanmean(bid_volumes[:min(length, len(bid_volumes))])
        densities[f"asks_{length}"] = np.nanmean(ask_volumes[:min(length, len(ask_volumes))])

    return densities
