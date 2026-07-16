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
