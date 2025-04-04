import os
import pandas as pd

def fill_starting_nan(series):
    """
    Fill leading NaNs with the first valid value to simulate holding the asset
    from the first available price onward.
    """
    first_valid = series.first_valid_index()
    if first_valid is not None:
        series.loc[:first_valid] = series.loc[first_valid]
    return series


def merge_price_data(root_dir, output_path="prices.parquet"):
    all_prices = []

    for coin_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, coin_folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith("_USD.parquet"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_parquet(file_path)
                        coin = file.replace("_USD.parquet", "")
                        df = df[["date", "close"]].copy()
                        df["date"] = pd.to_datetime(df["date"])

                        # Invert log_10(x)/10 transformation if applied
                        df["close"] = 10 ** (df["close"] * 10)

                        df = df.rename(columns={"close": coin})
                        df = df.set_index("date")
                        all_prices.append(df)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    # Outer join on all dates
    merged = pd.concat(all_prices, axis=1).sort_index()

    # Backfill leading NaNs with first known price
    merged = merged.apply(fill_starting_nan)

    # Forward-fill subsequent missing prices
    merged = merged.ffill()

    # Drop rows where all assets are still NaN (before any coin had data)
    merged = merged.dropna(how="all")

    # Save to Parquet
    merged.to_parquet(output_path)
    print(f"✅ Saved merged file: {output_path}")
    print(f"✔️  {len(merged.columns)} assets | 📅 {merged.index.min().date()} → {merged.index.max().date()} | 📈 {len(merged)} rows")

# Example usage
if __name__ == "__main__":
    merge_price_data('C:/Users/kfern/Desktop/Business/Crypto Site/App/Raw Data/')
