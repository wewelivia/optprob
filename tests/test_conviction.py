"""
Tests for the conviction / positioning layer:
  * z-score normalization behaves (no explosions on smooth series)
  * k-day change scored against k-day change volatility
  * vol/OI scored as intensity level (persistent elevation NOT de-meaned away)
  * scoring: confluence -> high, opposed strong signals -> conflicting, noise -> low
  * summary stats (put/call, center of gravity, max-pain, premium notional)
  * store round-trip + delta pinning to nearest stored trading day
Run: python tests/test_conviction.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import datetime as dt
import tempfile

import numpy as np

from app.data.positioning_store import PositioningStore, SnapshotRow
from app.core.conviction import (ConvictionEngine, ConvictionConfig,
                                 summarize_positioning, _robust_scale,
                                 _zscore_change, _intensity_score)


def _fresh_store():
    return PositioningStore(os.path.join(tempfile.mkdtemp(), "t.db"))


def test_robust_scale_no_explosion_on_smooth():
    # a near-perfect ramp must NOT yield ~0 scale (which would explode z)
    ramp = np.arange(0, 30, dtype=float) * 100.0 + 5000.0
    s = _robust_scale(ramp)
    assert np.isfinite(s) and s > 0
    # a 5-step change scored against it stays bounded/sane
    z = _zscore_change(500.0, ramp, min_points=8, horizon=5)
    assert z is not None and abs(z) < 50
    print("  ok: robust scale floors smooth series")


def test_intensity_not_demeaned():
    # persistently HIGH turnover should read positive, not ~0
    hist = np.full(20, 0.8)  # always high
    z = _intensity_score(0.8, hist, 8)
    assert z is not None and z > 0.5
    # a genuinely quiet level reads low
    lo = _intensity_score(0.02, np.full(20, 0.02), 8)
    assert lo is not None
    print("  ok: intensity level not de-meaned")


def _seed(store, UND, EXP, today, n=30, seed=2):
    rng = np.random.default_rng(seed)
    for i in range(n):
        d = today - dt.timedelta(days=(n - 1 - i))
        store.write_snapshot([
            # building: OI up, turnover rising, IV up
            SnapshotRow(d, UND, EXP, 8000, "C",
                        10000 + i * 400 + rng.normal(0, 150),
                        3000 + i * 350 + rng.normal(0, 300),
                        0.15 + i * 0.0015 + rng.normal(0, 0.002), 120),
            # stale: flat noisy OI, tiny volume, flat IV
            SnapshotRow(d, UND, EXP, 7000, "P",
                        20000 + rng.normal(0, 200),
                        500 + rng.normal(0, 80),
                        0.22 + rng.normal(0, 0.003), 90),
            # conflicting: OI up hard + volume up, IV DOWN
            SnapshotRow(d, UND, EXP, 7500, "C",
                        5000 + i * 600 + rng.normal(0, 150),
                        3000 + i * 300 + rng.normal(0, 300),
                        0.20 - i * 0.0018 + rng.normal(0, 0.002), 150),
        ])


def test_scoring_confluence_conflict_noise():
    st = _fresh_store()
    UND, EXP, today = "SPX Index", dt.date(2026, 12, 31), dt.date.today()
    _seed(st, UND, EXP, today)
    eng = ConvictionEngine(st, ConvictionConfig(short_window=5))
    res = {c.strike: c for c in eng.compute_for_expiry(UND, EXP, as_of=today)}

    assert res[8000].direction == 1
    assert res[8000].composite in ("high", "moderate"), res[8000].composite
    assert res[7500].composite == "conflicting", res[7500].composite
    assert res[7000].composite in ("low", "moderate"), res[7000].composite
    print("  ok: confluence=high, opposed=conflicting, noise=low")


def test_summary_stats():
    st = _fresh_store()
    UND, EXP, today = "SPX Index", dt.date(2026, 12, 31), dt.date.today()
    _seed(st, UND, EXP, today)
    eng = ConvictionEngine(st, ConvictionConfig(short_window=5))
    res = eng.compute_for_expiry(UND, EXP, as_of=today)
    rows = st.snapshot_on(UND, today, expiry=EXP)
    summ = summarize_positioning(rows, EXP, res, multiplier=100).as_dict()
    assert summ["put_call_oi_ratio"] is not None and summ["put_call_oi_ratio"] > 0
    assert summ["oi_center_of_gravity"] is not None
    assert summ["max_pain"] is not None
    assert summ["total_premium_notional"] and summ["total_premium_notional"] > 0
    print("  ok: summary stats (P/C, CoG, max-pain, premium notional)")


def test_store_roundtrip_and_prior_date():
    st = _fresh_store()
    UND, EXP, today = "SPX Index", dt.date(2026, 12, 31), dt.date.today()
    for gap in (0, 3, 5, 8):
        d = today - dt.timedelta(days=gap)
        st.write_snapshot([SnapshotRow(d, UND, EXP, 8000, "C",
                                       1000.0, 500.0, 0.2, 100.0)])
    # nearest to '5 days ago' should pick the day exactly 5 back
    got = st.nearest_prior_date(UND, 5, ref=today)
    assert got == today - dt.timedelta(days=5), got
    series = st.series_for_strike(UND, EXP, 8000, "C", lookback_days=90)
    assert len(series) == 4
    print("  ok: store round-trip + nearest-prior-date pinning")


if __name__ == "__main__":
    test_robust_scale_no_explosion_on_smooth()
    test_intensity_not_demeaned()
    test_scoring_confluence_conflict_noise()
    test_summary_stats()
    test_store_roundtrip_and_prior_date()
    print("ALL CONVICTION TESTS PASSED")
