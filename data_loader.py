"""
Download ATP match data from Jeff Sackmann's tennis_atp GitHub repository.
Data covers 2000–2024 and is used for training the prediction model.
"""

import os
import io
import requests
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch_year(year: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/atp_matches_{year}.csv"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception as e:
        print(f"  Could not fetch {year}: {e}")
        return None


def load_atp_matches(start: int = 2000, end: int = 2024) -> pd.DataFrame:
    cache_path = os.path.join(DATA_DIR, f"atp_matches_{start}_{end}.parquet")
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    frames = []
    for year in range(start, end + 1):
        print(f"  Fetching {year}...", end=" ")
        df = fetch_year(year)
        if df is not None:
            print(f"{len(df)} matches")
            frames.append(df)
        else:
            print("skipped")

    data = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    data.to_parquet(cache_path)
    print(f"Saved {len(data):,} matches to {cache_path}")
    return data
