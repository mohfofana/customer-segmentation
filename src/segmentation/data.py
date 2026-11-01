"""Load the marts exported by the ecommerce-data-pipeline project."""
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/mohfofana/ecommerce-data-pipeline/main/exports/"
DATA = Path(__file__).resolve().parents[2] / "data"


def _load(name: str) -> pd.DataFrame:
    path = DATA / f"{name}.parquet"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        pd.read_parquet(BASE + f"{name}.parquet").to_parquet(path)
    return pd.read_parquet(path)


def load_rfm() -> pd.DataFrame:
    return _load("mart_customer_rfm")


def load_first_orders(observation_end: str = "2018-03-01") -> pd.DataFrame:
    """First order of each customer, with a flag telling if they ever came back.

    Only customers whose first order happened before `observation_end` are kept,
    so that everyone had at least ~6 months to place a second order. Without
    this, recent customers would look 'not returning' just because the data stops.
    """
    orders = _load("fct_orders")
    orders = orders[~orders["order_status"].isin(["canceled", "unavailable"])]
    orders = orders.sort_values("purchased_at")
    orders["order_rank"] = orders.groupby("customer_unique_id").cumcount()
    n_orders = orders.groupby("customer_unique_id").size()
    first = orders[orders["order_rank"] == 0].copy()
    first["returned"] = first["customer_unique_id"].map(n_orders).gt(1).astype(int)
    return first[first["purchased_at"] < observation_end].reset_index(drop=True)
