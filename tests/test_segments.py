import pandas as pd

from segmentation.segments import ORDER, assign_segments


def _rfm(**kw):
    base = dict(customer_unique_id="c", frequency=1, r_score=3, m_score=3,
                avg_review_score=5.0, recency_days=200, monetary=100.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_every_customer_gets_exactly_one_known_segment():
    df = pd.concat([_rfm(frequency=f, r_score=r, m_score=m, avg_review_score=s)
                    for f in (1, 2) for r in (1, 5) for m in (1, 5) for s in (1, 5)], ignore_index=True)
    seg = assign_segments(df)
    assert seg.notna().all()
    assert set(seg.astype(str)) <= set(ORDER)


def test_rules_priority():
    assert assign_segments(_rfm(frequency=3, r_score=5))[0] == "Fidèles actifs"
    assert assign_segments(_rfm(frequency=2, r_score=1))[0] == "Fidèles à réactiver"
    # an unhappy one-time buyer is "Déçus" even if recent and high value
    assert assign_segments(_rfm(avg_review_score=1, r_score=5, m_score=5))[0] == "Déçus"
    assert assign_segments(_rfm(r_score=5, m_score=5))[0] == "Nouveaux gros paniers"
    assert assign_segments(_rfm(r_score=1, m_score=5))[0] == "Gros paniers endormis"
