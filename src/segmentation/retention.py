"""What makes a first-time customer come back?"""
import numpy as np
import pandas as pd
from scipy import stats


def return_rate_by(first: pd.DataFrame, col: str, min_size: int = 500) -> pd.DataFrame:
    g = first.groupby(col, observed=True)["returned"].agg(["mean", "size"])
    g = g[g["size"] >= min_size].rename(columns={"mean": "return_rate", "size": "customers"})
    return g.sort_values("return_rate", ascending=False)


def compare_rates(a: pd.Series, b: pd.Series) -> dict:
    """Two-proportion z-test between groups a and b (binary series)."""
    n1, n2 = len(a), len(b)
    p1, p2 = a.mean(), b.mean()
    p = (a.sum() + b.sum()) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return {"rate_a": p1, "rate_b": p2, "relative_change": p1 / p2 - 1,
            "z": z, "p_value": 2 * (1 - stats.norm.cdf(abs(z))), "n_a": n1, "n_b": n2}
