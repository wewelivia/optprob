"""
Rate-space transform for IMM-style rate futures (SOFR / fed funds / Euribor).

These contracts quote as (100 - rate) and the options are struck on that PRICE.
The dashboard fits SABR and runs Breeden-Litzenberger in price space -- which is
where the market and its quoted vols live -- then maps the finished density to
rate space so a strategist can ask "above 4%" rather than translating to
"below 96" by hand.

The load-bearing property is the change-of-variable identity:

    R = ref - P,  |dP/dR| = 1
    q_R(r) = q_P(ref - r)
    P(R > r) == P(P < ref - r)

If that last line ever fails, the dashboard is quietly answering the opposite
question. Everything else here is secondary.

Run:  python tests/test_rate_space.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "backend"))

import numpy as np

from app.core.breeden_litzenberger import extract_rnd, to_rate_space
from app.core.sabr import calibrate_sabr
from app.data.bloomberg import (ASSET_DEFAULTS, RATE_FUTURE_REF, MockProvider,
                                is_rate_future)
from app.data.chain_builder import classify_asset

REF = RATE_FUTURE_REF


def _price_rnd():
    """A price-space density on a synthetic Dec-26 SOFR future (~96.05)."""
    chain = MockProvider().build_chain("SFRZ6 Comdty", n_expiries=3)
    sl = chain.expiries[-1]
    strikes, vols = sl.smile()
    params = calibrate_sabr(sl.forward, strikes, vols, sl.T,
                            beta=ASSET_DEFAULTS["RATES_PRICE"]["beta"],
                            shift=chain.shift)
    rnd = extract_rnd(params, r=0.0, strike_lo=float(strikes.min()) - 3.0,
                      strike_hi=float(strikes.max()) + 3.0, n_grid=1200)
    return chain, sl, rnd


# ---------------------------------------------------------------------------
def test_classification():
    # Real rate futures.
    for t in ("SFRZ6 Comdty", "FFZ6 Comdty", "SFRH7 Comdty", "ERM7 Comdty",
              "sfrz26 comdty"):
        assert is_rate_future(t), t
        assert classify_asset(t) == "RATES_PRICE", t

    # Must NOT be swallowed: IMM-month-coded commodity tickers are the exact
    # false-positive risk, since a mistaken match mirrors the density about 100.
    for t in ("CLZ6 Comdty", "GCZ6 Comdty", "CL1 Comdty", "XAU Curncy",
              "SPX Index", "AAPL US Equity", "FEDFUNDS", "SOFR"):
        assert not is_rate_future(t), t
        assert classify_asset(t) != "RATES_PRICE", t

    # The rate-in-percent presets stay on the old RATES path.
    assert classify_asset("FEDFUNDS") == "RATES"
    print("  ok: rate-future detection (and no commodity false positives)")


def test_transform_identity():
    """The load-bearing test: same question, both spaces, same answer."""
    _, _, rp = _price_rnd()
    rr = to_rate_space(rp, ref=REF)

    for rate in (3.00, 3.50, 3.75, 4.00, 4.25, 4.50):
        price = REF - rate
        got = rr.prob_above(rate)
        want = rp.prob_below(price)
        assert abs(got - want) < 1e-9, f"P(rate>{rate}) {got} != P(price<{price}) {want}"

        got_b = rr.prob_below(rate)
        want_b = rp.prob_above(price)
        assert abs(got_b - want_b) < 1e-9, f"below mismatch at {rate}"
    print("  ok: P(rate > r) == P(price < ref - r) across the grid")


def test_transform_wellformed():
    _, sl, rp = _price_rnd()
    rr = to_rate_space(rp, ref=REF)

    # Grid ascends (quantile's np.interp on the CDF needs monotonicity).
    assert np.all(np.diff(rr.strikes) > 0), "rate grid not ascending"
    assert np.all(np.diff(rr.cdf) >= -1e-12), "rate cdf not monotone"

    # Density still integrates to 1 and is non-negative.
    area = np.trapezoid(rr.pdf, rr.strikes)
    assert abs(area - 1.0) < 1e-6, f"pdf area {area}"
    assert np.all(rr.pdf >= 0.0)

    # CDF spans [0, 1].
    assert abs(rr.cdf[0]) < 1e-9 and abs(rr.cdf[-1] - 1.0) < 1e-9

    # Forward maps.
    assert abs(rr.F - (REF - rp.F)) < 1e-12
    assert abs(rr.F - (REF - sl.forward)) < 1e-9

    # Mean maps (affine change of variable).
    assert abs(rr.mean() - (REF - rp.mean())) < 1e-6, "mean did not map"

    # Quantiles mirror: rate p05 == ref - price p95.
    assert abs(rr.quantile(0.05) - (REF - rp.quantile(0.95))) < 1e-3
    assert abs(rr.quantile(0.95) - (REF - rp.quantile(0.05))) < 1e-3

    # T and r pass through untouched.
    assert rr.T == rp.T and rr.r == rp.r

    # Involution: mapping twice returns the original.
    back = to_rate_space(rr, ref=REF)
    assert np.allclose(back.strikes, rp.strikes)
    assert np.allclose(back.pdf, rp.pdf)
    assert np.allclose(back.cdf, rp.cdf)
    print("  ok: rate density well-formed, maps forward/mean/quantiles, involutive")


def test_end_to_end():
    from app.core import service

    out = service.compute_distribution("SFRZ6 Comdty",
                                       "above 4% by December",
                                       prefer_live=False)
    assert out["asset_class"] == "RATES_PRICE"
    assert out["rate_space"] is True
    assert out["is_percent"] is True
    assert out["rate_future_ref"] == REF

    # Forward reported as a RATE, not a price.
    assert 2.0 < out["forward"] < 6.0, f"forward {out['forward']} not rate-like"
    assert abs(out["forward"] - (REF - out["forward_price_space"])) < 1e-9

    # Grid is in rate space and ascends.
    grid = np.array(out["grid"])
    assert np.all(np.diff(grid) > 0)
    assert grid.min() > -5.0 and grid.max() < 12.0, "grid not rate-scaled"

    # Probability is a real number in (0, 1) and the complement agrees.
    p = out["probability"]
    assert 0.0 < p < 1.0, p
    assert abs(out["complement"] - (1.0 - p)) < 1e-12
    assert out["direction"] == "above" and out["threshold"] == 4.0

    # Stats came back in rate space.
    assert 2.0 < out["stats"]["median"] < 6.0

    # Smile x-axis mapped to rates, ascending, price space retained.
    xs = [s["strike"] for s in out["smile"]]
    assert xs == sorted(xs)
    for s in out["smile"]:
        assert abs(s["strike"] - (REF - s["strike_price_space"])) < 1e-9

    # Directional sanity: forward ~3.95%, so P(above 4%) should sit near a
    # coin flip, and P(above) must fall as the threshold rises.
    ps = [service.compute_distribution("SFRZ6 Comdty", f"above {t}% by December",
                                       prefer_live=False)["probability"]
          for t in (3.0, 3.5, 4.0, 4.5, 5.0)]
    assert all(a > b for a, b in zip(ps, ps[1:])), f"not monotone: {ps}"
    print(f"  ok: end-to-end SFRZ6 fwd={out['forward']:.3f}% "
          f"P(>4%)={p:.3f} monotone={[round(x, 3) for x in ps]}")


def test_non_rate_unaffected():
    """Regression: the existing paths must be untouched."""
    from app.core import service
    out = service.compute_distribution("SPX Index", "above 8000 by December",
                                       prefer_live=False)
    assert out["rate_space"] is False
    assert out["forward_price_space"] is None
    assert out["rate_future_ref"] is None
    assert out["asset_class"] == "EQ_INDEX"
    assert 0.0 <= out["probability"] <= 1.0
    for s in out["smile"]:
        assert s["strike"] == s["strike_price_space"]

    out = service.compute_distribution("FEDFUNDS", "above 4% by December",
                                       prefer_live=False)
    assert out["asset_class"] == "RATES" and out["rate_space"] is False
    assert out["is_percent"] is True
    print("  ok: SPX / FEDFUNDS paths unaffected")


test_classification()
test_transform_identity()
test_transform_wellformed()
test_end_to_end()
test_non_rate_unaffected()
print("ALL RATE-SPACE TESTS PASSED")


# ---------------------------------------------------------------------------
# Backfill on FUTURES options.
#
# Regression cover for the live failure:
#   "Backfill failed: securities is required for HistoricalDataRequest"
# Cause: backfill_history_rows built `meta` purely by parsing tickers.
# _parse_opt_ticker expects an equity-style 'MM/DD/YY C<strike>' ticker; a
# futures option ('SFRZ6C 96.00 Comdty') carries NO expiry in the string, so
# every parse returned None, meta was {}, and bdh() was called with an empty
# securities list. There was a guard for empty `members` but not for empty
# `meta`.
# ---------------------------------------------------------------------------
class _FakeFuturesBBG:
    """Bloomberg stand-in returning SOFR-option tickers and their BDP fields."""

    EXPIRY = __import__("datetime").date(2026, 12, 11)

    def __init__(self):
        self.strikes = [95.50, 95.75, 96.00, 96.25, 96.50]
        self.tickers = [f"SFRZ6{cp} {k:.2f} Comdty"
                        for k in self.strikes for cp in ("C", "P")]
        self.bdh_calls = []

    def chain_tickers(self, ticker, call_put="C"):
        return list(self.tickers)

    def option_fields(self, tickers):
        import pandas as pd
        rows = []
        for tk in tickers:
            cp = tk.split()[0][-1]
            k = float(tk.split()[1])
            rows.append({"opt_strike_px": k, "opt_expire_dt": self.EXPIRY,
                         "opt_put_call": cp})
        return pd.DataFrame(rows, index=list(tickers))

    def bdh_fields(self, tickers, flds, start, end):
        import datetime as _dt
        import pandas as pd
        self.bdh_calls.append(list(tickers))
        if not tickers:
            # Mirror blpapi's real behaviour so the test fails the same way.
            raise Exception("securities is required for HistoricalDataRequest")
        days = [end - _dt.timedelta(days=i) for i in range(4)]
        rec = []
        for tk in tickers:
            for d in days:
                rec += [{"date": d, "ticker": tk, "field": "open_int", "value": 1200.0},
                        {"date": d, "ticker": tk, "field": "ivol_mid", "value": 0.012},
                        {"date": d, "ticker": tk, "field": "px_last", "value": 0.15},
                        {"date": d, "ticker": tk, "field": "px_volume", "value": 300.0}]
        return pd.DataFrame(rec)


def test_backfill_futures_options():
    from app.data import chain_builder as cb

    bbg = _FakeFuturesBBG()

    # The ticker parser genuinely cannot do this -- confirm the premise.
    assert all(cb._parse_opt_ticker(t) is None for t in bbg.tickers), \
        "futures tickers unexpectedly parseable; test premise is stale"

    rows = cb.backfill_history_rows(bbg, "SFRZ6 Comdty", days=30,
                                    max_options=100)
    assert rows, "no rows produced"
    # bdh must never be handed an empty securities list.
    assert all(c for c in bbg.bdh_calls), "bdh called with empty securities"

    # Metadata resolved via BDP, not the ticker string.
    assert {r.strike for r in rows} == set(bbg.strikes)
    assert {r.call_put for r in rows} == {"C", "P"}
    assert {r.expiry for r in rows} == {_FakeFuturesBBG.EXPIRY}
    assert all(r.underlying == "SFRZ6 Comdty" for r in rows)
    assert all(r.open_interest == 1200.0 for r in rows)
    print(f"  ok: futures-option backfill via BDP -> {len(rows)} rows, "
          f"{len({r.strike for r in rows})} strikes")


def test_backfill_guard_raises_actionably():
    """Unresolvable metadata must raise something a human can act on, not
    blpapi's opaque 'securities is required'."""
    from app.data import chain_builder as cb

    class _Blind(_FakeFuturesBBG):
        def option_fields(self, tickers):
            import pandas as pd
            return pd.DataFrame()   # BDP resolves nothing

    bbg = _Blind()
    try:
        cb.backfill_history_rows(bbg, "SFRZ6 Comdty", days=30, max_options=100)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        msg = str(e)
        assert "Could not resolve strike/expiry" in msg
        assert "SFRZ6" in msg
    assert not any(c == [] for c in bbg.bdh_calls), "bdh reached with empty list"
    print("  ok: unresolvable metadata raises actionably before touching bdh")


def test_underlying_whitespace_stripped():
    """A trailing space from a UI field must not become part of the store key."""
    from app.core import service
    a = service.compute_distribution("SFRZ6 Comdty ", "above 4% by December",
                                     prefer_live=False)
    b = service.compute_distribution("SFRZ6 Comdty", "above 4% by December",
                                     prefer_live=False)
    assert a["underlying"] == b["underlying"] == "SFRZ6 Comdty"
    assert a["asset_class"] == "RATES_PRICE"
    assert abs(a["probability"] - b["probability"]) < 1e-12
    print("  ok: trailing whitespace stripped from underlying")


test_backfill_futures_options()
test_backfill_guard_raises_actionably()
test_underlying_whitespace_stripped()
print("ALL FUTURES-BACKFILL TESTS PASSED")
