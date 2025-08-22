
import os
import pandas as pd

def split_walk_forward(prices: pd.DataFrame, train_frac: float = 0.7):
    n = len(prices)
    cut = max(1, int(n * train_frac))
    return prices.iloc[:cut].copy(), prices.iloc[cut:].copy()

def load_builtin_sample() -> pd.DataFrame:
    here = os.path.dirname(__file__)
    p = os.path.join(here, "sample_prices.csv")
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()
