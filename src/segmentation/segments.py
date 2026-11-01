"""Business segments, readable by a marketing team.

Classic RFM grids assume customers buy several times. Here 97 % of customers
bought once, so frequency barely varies. Segments therefore combine recency,
value, repeat behaviour and satisfaction: the levers a CRM team can act on.
"""
import numpy as np
import pandas as pd

SEGMENTS = {
    "Fidèles actifs": "2+ commandes, dernier achat dans les 40 % les plus récents",
    "Fidèles à réactiver": "2+ commandes, mais plus d'achat depuis longtemps",
    "Déçus": "une seule commande, note moyenne de 1 ou 2",
    "Nouveaux gros paniers": "une commande récente, dans les 20 % des paniers les plus élevés",
    "Nouveaux clients": "une commande récente",
    "Gros paniers endormis": "une commande ancienne, dans les 20 % des paniers les plus élevés",
    "Endormis": "une commande ancienne, panier standard",
}

ORDER = list(SEGMENTS)


def assign_segments(rfm: pd.DataFrame) -> pd.Series:
    repeat = rfm["frequency"] >= 2
    recent = rfm["r_score"] >= 4            # 40 % most recent
    high_value = rfm["m_score"] == 5        # top 20 % basket value
    unhappy = rfm["avg_review_score"] <= 2

    conditions = [
        repeat & recent,
        repeat & ~recent,
        ~repeat & unhappy,
        ~repeat & recent & high_value,
        ~repeat & recent,
        ~repeat & ~recent & high_value,
        ~repeat & ~recent,
    ]
    seg = np.select(conditions, ORDER, default="Endormis")
    return pd.Series(pd.Categorical(seg, categories=ORDER, ordered=True), index=rfm.index, name="segment")


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    total_rev = rfm["monetary"].sum()
    out = (rfm.groupby("segment", observed=True)
              .agg(customers=("customer_unique_id", "size"),
                   revenue=("monetary", "sum"),
                   avg_basket=("monetary", "mean"),
                   avg_recency_days=("recency_days", "mean"),
                   avg_orders=("frequency", "mean"),
                   avg_review=("avg_review_score", "mean"),
                   late_rate=("late_delivery_rate", "mean")))
    out["share_customers"] = out["customers"] / out["customers"].sum()
    out["share_revenue"] = out["revenue"] / total_rev
    return out
