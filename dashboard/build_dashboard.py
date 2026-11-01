"""Build the interactive segmentation dashboard (single static HTML page).

    python dashboard/build_dashboard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segmentation.data import load_first_orders, load_rfm  # noqa: E402
from segmentation.retention import compare_rates, return_rate_by  # noqa: E402
from segmentation.segments import SEGMENTS, assign_segments, segment_summary  # noqa: E402

INK, SIGNAL, SLATE, LINE, BG, GREY = "#0F3D5C", "#E4572E", "#5B6B7A", "#D8DFE6", "#FFFFFF", "#B8C4CF"
FONT = "Archivo, system-ui, sans-serif"


def style(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=BG,
                      plot_bgcolor=BG, font=dict(family=FONT, size=13, color="#1F2933"),
                      hoverlabel=dict(font_family=FONT, bgcolor="white"), showlegend=False)
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#EEF1F4", zeroline=False)
    return fig


def html(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displayModeBar": False, "responsive": True})


def fr(v, fmt):
    return format(v, fmt).replace(",", " ").replace(".", ",")


def main():
    rfm = load_rfm()
    rfm["segment"] = assign_segments(rfm)
    s = segment_summary(rfm).sort_values("share_revenue")
    rules = [SEGMENTS[i] for i in s.index]

    f1 = go.Figure()
    f1.add_bar(y=s.index, x=s.share_customers, orientation="h", name="part des clients", marker_color=GREY,
               customdata=np.stack([s.customers, rules], axis=1),
               hovertemplate="<b>%{y}</b><br>%{x:.1%} des clients (%{customdata[0]:,})<br>%{customdata[1]}<extra></extra>")
    f1.add_bar(y=s.index, x=s.share_revenue, orientation="h", name="part du CA", marker_color=INK,
               customdata=s.avg_basket,
               hovertemplate="<b>%{y}</b><br>%{x:.1%} du chiffre d'affaires<br>panier moyen R$ %{customdata:.0f}<extra></extra>")
    f1.update_layout(barmode="group", bargap=0.25, showlegend=True)
    style(f1, 420)
    f1.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12, x=0, traceorder="normal"))
    f1.update_xaxes(tickformat=".0%")

    first = load_first_orders()
    test = compare_rates(first[first.is_late == True].returned, first[first.is_late == False].returned)  # noqa: E712
    rates = first.dropna(subset=["is_late"]).groupby("is_late").returned.agg(["mean", "size"])
    err = 1.96 * np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["size"])
    f2 = go.Figure(go.Bar(x=["livrée à l'heure", "livrée en retard"], y=rates["mean"],
                          error_y=dict(type="data", array=err, color=SLATE, thickness=1.5),
                          marker_color=[INK, SIGNAL], customdata=rates["size"],
                          hovertemplate="%{x}<br>%{y:.2%} reviennent<br>%{customdata:,} clients<extra></extra>"))
    f2.update_yaxes(tickformat=".1%")
    style(f2)

    cat = return_rate_by(first, "main_category", 2000).head(12).sort_values("return_rate")
    avg = first.returned.mean()
    f3 = go.Figure(go.Bar(y=cat.index.str.replace("_", " "), x=cat.return_rate, orientation="h",
                          marker_color=[SIGNAL if v < avg else INK for v in cat.return_rate],
                          customdata=cat.customers,
                          hovertemplate="%{y}<br>%{x:.1%} reviennent<br>%{customdata:,} clients<extra></extra>"))
    f3.add_vline(x=avg, line=dict(color=SLATE, dash="dot"))
    f3.update_xaxes(tickformat=".0%")
    style(f3, 360)

    orders = pd.read_parquet(ROOT / "data" / "fct_orders.parquet")
    orders = orders[~orders.order_status.isin(["canceled", "unavailable"])].sort_values("purchased_at")
    orders["rank"] = orders.groupby("customer_unique_id").cumcount()
    t = orders.pivot_table(index="customer_unique_id", columns="rank", values="purchased_at", aggfunc="first")
    gap = (t[1] - t[0]).dropna().dt.days
    f4 = go.Figure(go.Histogram(x=gap.clip(upper=360), xbins=dict(start=0, end=370, size=10),
                                marker_color=INK, hovertemplate="%{x} jours<br>%{y} clients<extra></extra>"))
    f4.update_xaxes(title_text="jours entre la 1re et la 2e commande (plafonné à 360)")
    style(f4, 280)

    repeat_share = (rfm.frequency > 1).mean()
    kpis = [
        (fr(len(rfm), ",d"), "clients analysés"),
        (fr(1 - repeat_share, ".1%").replace("%", " %"), "n'ont commandé qu'une fois"),
        (fr(test["relative_change"] * 100, "+.0f") + " %", "de retour après une 1re livraison en retard"),
        ("x" + fr(cat.return_rate.max() / cat.return_rate.min(), ".1f"), "d'écart de fidélisation entre catégories"),
    ]
    out = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    for i, (v, label) in enumerate(kpis, 1):
        out = out.replace(f"{{{{kpi{i}_value}}}}", v).replace(f"{{{{kpi{i}_label}}}}", label)
    for name, fig in {"segments": f1, "late": f2, "category": f3, "gap": f4}.items():
        out = out.replace(f"{{{{{name}}}}}", html(fig, name))
    (ROOT / "dashboard" / "index.html").write_text(out, encoding="utf-8")
    print("dashboard/index.html written")


if __name__ == "__main__":
    main()
