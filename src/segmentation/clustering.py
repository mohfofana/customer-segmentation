"""Unsupervised check of the business segments with K-means."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["recency_days", "log_monetary", "frequency", "avg_review_score", "late_delivery_rate"]


def prepare(rfm: pd.DataFrame) -> pd.DataFrame:
    X = rfm.copy()
    X["log_monetary"] = np.log1p(X["monetary"])
    X["frequency"] = X["frequency"].clip(upper=4)
    X["avg_review_score"] = X["avg_review_score"].fillna(X["avg_review_score"].median())
    X["late_delivery_rate"] = X["late_delivery_rate"].fillna(0)
    return X[FEATURES]


def choose_k(X: pd.DataFrame, ks=range(3, 9), sample: int = 15000, seed: int = 0) -> pd.DataFrame:
    Z = StandardScaler().fit_transform(X)
    idx = np.random.default_rng(seed).choice(len(Z), size=min(sample, len(Z)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z[idx], km.labels_[idx])})
    return pd.DataFrame(rows)


def fit_clusters(X: pd.DataFrame, k: int, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=X.columns)
    centers["monetary"] = np.expm1(centers.pop("log_monetary"))
    return km.labels_, centers
