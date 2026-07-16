"""
Conviction / positioning signals engine.

Turns raw strike-level history (OI, volume, IV) into a per-strike conviction
read that distinguishes *genuine building conviction* from noise and mechanical
flow. The design follows a specific philosophy:

THREE SIGNALS, MUST CORROBORATE
-------------------------------
  * base layer     : OI level + change in OI      -> is a position building?
  * intensity      : volume / OI ratio            -> is today's flow large vs the
                                                     existing stock (fresh vs stale)?
  * pricing confirm: IV level + change in IV       -> are people paying up for it?

A strike where OI is building AND volume/OI is elevated AND IV is rising is a far
stronger read than any single metric. One moving alone is noise; several aligned
is a real position being put on with urgency and price-insensitivity.

Z-SCORING IS THE CRUX
---------------------
A raw 5-day OI change means very different things in a quiet summer month vs.
around an FOMC meeting. So every *change* signal is normalized against the
**60-day rolling volatility of that signal's own series** at that strike. This
kills false positives during naturally choppy periods.

SCORING -- THREE SEPARATE OUTPUTS (not one blended number)
----------------------------------------------------------
  1. magnitude score : weighted sum of the z-scores. IV-change weighted highest
                       (the market's own price on the view), volume/OI second
                       (fresh vs stale), raw OI change lowest (most contaminated
                       by mechanical/rebalancing flow).
  2. agreement flag  : how many of the three point the SAME direction -- shown
                       alongside, NOT folded into the magnitude.
  3. composite read  : magnitude GATED by agreement. High-conviction only if the
                       magnitude is large AND >=2 of 3 agree on direction. Large
                       magnitude with disagreement -> a separate 'conflicting'
                       flag rather than a misleadingly moderate average.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.data.positioning_store import PositioningStore, SnapshotRow


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ConvictionConfig:
    # windows (calendar days; snapped to the nearest stored trading day)
    short_window: int = 5      # operational signal you act on
    trend_window: int = 20     # trend confirmation
    context_window: int = 60   # z-score normalization horizon

    # magnitude weights -- IV change trusted most, OI change least.
    w_iv: float = 0.45
    w_voloi: float = 0.35
    w_oi: float = 0.20

    # thresholds
    high_conviction_magnitude: float = 1.0   # |magnitude| gate for a flag
    min_agreement: int = 2                    # >=2 of 3 signals must agree
    conflict_magnitude: float = 1.0           # large magnitude but disagreeing
    min_context_points: int = 8               # need this many obs to z-score


@dataclass
class StrikeConviction:
    expiry: dt.date
    strike: float
    call_put: str
    # raw levels / deltas
    oi: Optional[float]
    d_oi_short: Optional[float]
    vol_oi_ratio: Optional[float]
    iv: Optional[float]
    d_iv_short: Optional[float]
    # z-scores (normalized against own 60d vol)
    z_oi: Optional[float]
    z_voloi: Optional[float]
    z_iv: Optional[float]
    # outputs
    magnitude: Optional[float]
    agreement: int              # -3..+3 net direction count (signed)
    n_agree: int                # count agreeing with dominant direction
    direction: int              # +1 / -1 / 0 dominant direction
    composite: str              # 'high' | 'moderate' | 'low' | 'conflicting' | 'na'

    def as_dict(self) -> dict:
        return {
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "call_put": self.call_put,
            "oi": _num(self.oi),
            "d_oi_short": _num(self.d_oi_short),
            "vol_oi_ratio": _num(self.vol_oi_ratio),
            "iv": _num(self.iv),
            "d_iv_short": _num(self.d_iv_short),
            "z_oi": _num(self.z_oi),
            "z_voloi": _num(self.z_voloi),
            "z_iv": _num(self.z_iv),
            "magnitude": _num(self.magnitude),
            "agreement": self.agreement,
            "n_agree": self.n_agree,
            "direction": self.direction,
            "composite": self.composite,
        }


# ---------------------------------------------------------------------------
# Core stats helpers
# ---------------------------------------------------------------------------
def _num(x):
    if x is None:
        return None
    try:
        xf = float(x)
        return None if (xf != xf or math.isinf(xf)) else xf
    except (TypeError, ValueError):
        return None


def _robust_scale(series: np.ndarray) -> float:
    """Volatility of a signal's own series, used to normalize a change.

    Uses the std of first-differences (how much this series usually moves
    period-to-period), with a MAD-based fallback for robustness to outliers.
    Returns a positive scale, or nan if it can't be estimated.

    A FLOOR is applied relative to the signal's own level: a series that is a
    near-perfect linear ramp has almost-constant diffs (std ~ 0), which would
    otherwise produce explosive z-scores. We floor the scale at a small
    fraction of the typical level so a monotonic-but-smooth series can't yield
    an infinite z. This is the difference between 'unusually large move' and
    'this series simply always moves by roughly this much'.
    """
    s = series[np.isfinite(series)]
    if s.size < 3:
        return float("nan")
    diffs = np.diff(s)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return float("nan")
    sd = float(np.std(diffs, ddof=1))
    mad = float(np.median(np.abs(s - np.median(s)))) * 1.4826

    # floor: max of a fraction of the level scale and the MAD of levels.
    level = float(np.median(np.abs(s)))
    floor = max(1e-9, 0.02 * level, 0.25 * mad if np.isfinite(mad) else 0.0)

    best = max(sd, mad if np.isfinite(mad) else 0.0)
    best = max(best, floor)
    return best if best > 0 else float("nan")


def _zscore_change(recent_change: float, series: np.ndarray,
                   min_points: int, horizon: int = 1) -> Optional[float]:
    """Z-score a k-day change against the volatility of k-day changes in the
    signal's own series (NOT 1-day diffs).

    This is important: a steady multi-day build is unremarkable against daily
    noise but very remarkable against the distribution of same-horizon moves.
    We measure how the series typically moves over `horizon` steps and judge
    the observed change against that.
    """
    if recent_change is None or not np.isfinite(recent_change):
        return None
    s = series[np.isfinite(series)]
    if s.size < min_points:
        return None
    h = max(1, int(horizon))
    if s.size > h:
        hdiffs = s[h:] - s[:-h]           # all overlapping h-step changes
        hdiffs = hdiffs[np.isfinite(hdiffs)]
    else:
        hdiffs = np.array([], float)
    if hdiffs.size >= 2:
        sd = float(np.std(hdiffs, ddof=1))
        mad = float(np.median(np.abs(hdiffs - np.median(hdiffs)))) * 1.4826
        level = float(np.median(np.abs(s)))
        scale = max(sd, mad, 1e-9, 0.02 * level)
    else:
        scale = _robust_scale(series)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return float(recent_change / scale)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ConvictionEngine:
    def __init__(self, store: PositioningStore, cfg: ConvictionConfig | None = None):
        self.store = store
        self.cfg = cfg or ConvictionConfig()

    def _pick_prior_date(self, underlying: str, gap: int,
                         ref: dt.date) -> dt.date | None:
        return self.store.nearest_prior_date(underlying, gap, ref=ref)

    def compute_for_expiry(self, underlying: str, expiry: dt.date,
                           as_of: dt.date | None = None
                           ) -> list[StrikeConviction]:
        cfg = self.cfg
        as_of = as_of or dt.date.today()

        today_rows = self.store.snapshot_on(underlying, as_of, expiry=expiry)
        if not today_rows:
            # maybe the latest stored date isn't exactly today
            dates = self.store.available_dates(underlying, before=as_of)
            if not dates:
                return []
            as_of = dates[-1]
            today_rows = self.store.snapshot_on(underlying, as_of, expiry=expiry)

        prior_date = self._pick_prior_date(underlying, cfg.short_window, as_of)
        prior_rows = (self.store.snapshot_on(underlying, prior_date, expiry=expiry)
                      if prior_date and prior_date != as_of else [])
        prior_map = {(r.strike, r.call_put): r for r in prior_rows}

        out: list[StrikeConviction] = []
        for r in today_rows:
            key = (r.strike, r.call_put)
            prior = prior_map.get(key)

            # --- raw deltas (from our own complete local history) ---
            d_oi = (_delta(r.open_interest, prior.open_interest) if prior else None)
            d_iv = (_delta(r.implied_vol, prior.implied_vol) if prior else None)
            vol_oi = _safe_ratio(r.volume, r.open_interest)

            # --- context series for z-scoring (this strike's own history) ---
            series = self.store.series_for_strike(
                underlying, expiry, r.strike, r.call_put,
                lookback_days=cfg.context_window)
            oi_hist = np.array([_g(s.open_interest) for s in series], float)
            iv_hist = np.array([_g(s.implied_vol) for s in series], float)
            voloi_hist = np.array(
                [_safe_ratio(s.volume, s.open_interest) or float("nan")
                 for s in series], float)

            z_oi = _zscore_change(d_oi, oi_hist, cfg.min_context_points,
                                  horizon=cfg.short_window)
            z_iv = _zscore_change(d_iv, iv_hist, cfg.min_context_points,
                                  horizon=cfg.short_window)
            # vol/OI is an INTENSITY LEVEL signal: a persistently high turnover
            # ratio *is* the conviction signal, so we score how elevated today's
            # level is relative to a low-turnover baseline -- NOT relative to its
            # own recent mean (which would de-mean away a sustained elevation).
            z_voloi = _intensity_score(vol_oi, voloi_hist,
                                       cfg.min_context_points)

            sc = self._score(r, expiry, d_oi, d_iv, vol_oi, z_oi, z_voloi, z_iv)
            out.append(sc)
        out.sort(key=lambda s: s.strike)
        return out

    def _score(self, r: SnapshotRow, expiry: dt.date,
               d_oi, d_iv, vol_oi, z_oi, z_voloi, z_iv) -> StrikeConviction:
        cfg = self.cfg
        comps = [(z_iv, cfg.w_iv), (z_voloi, cfg.w_voloi), (z_oi, cfg.w_oi)]
        present = [(z, w) for z, w in comps if z is not None and np.isfinite(z)]

        if not present:
            return StrikeConviction(
                expiry, r.strike, r.call_put, r.open_interest, d_oi, vol_oi,
                r.implied_vol, d_iv, z_oi, z_voloi, z_iv,
                None, 0, 0, 0, "na")

        # weighted magnitude (re-normalize weights over present signals)
        wsum = sum(w for _, w in present)
        magnitude = sum(z * w for z, w in present) / wsum if wsum else None

        # direction agreement -- signed count over present signals
        signs = [(1 if z > 0 else (-1 if z < 0 else 0)) for z, _ in present]
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        if pos > neg:
            direction, n_agree = 1, pos
        elif neg > pos:
            direction, n_agree = -1, neg
        else:
            direction, n_agree = 0, max(pos, neg)
        agreement = pos - neg

        present_z = [z for z, _ in present]
        composite = self._composite(magnitude, n_agree, len(present), present_z)
        return StrikeConviction(
            expiry, r.strike, r.call_put, r.open_interest, d_oi, vol_oi,
            r.implied_vol, d_iv, z_oi, z_voloi, z_iv,
            magnitude, agreement, n_agree, direction, composite)

    def _composite(self, magnitude, n_agree, n_present, present_z) -> str:
        """Classify a strike into high / moderate / low / conflicting / na.

        CONFLICT is detected from the RAW signals, not the netted magnitude:
        genuine conflict is precisely when two strong signals point opposite
        ways and CANCEL in the weighted sum. Relying on |magnitude| would hide
        exactly those cases (e.g. OI building hard while IV falls). So we flag
        conflicting whenever at least two signals are individually strong and
        disagree in direction -- a large magnitude on its own is not required.
        """
        cfg = self.cfg
        if magnitude is None or not np.isfinite(magnitude):
            return "na"
        zs = [z for z in present_z if z is not None and np.isfinite(z)]
        strong = [z for z in zs if abs(z) >= cfg.conflict_magnitude]
        has_pos = any(z > 0 for z in strong)
        has_neg = any(z < 0 for z in strong)
        # Two or more strong signals pointing opposite ways -> conflicting.
        if n_present >= 2 and has_pos and has_neg:
            return "conflicting"

        mag = abs(magnitude)
        agree_ok = n_agree >= min(cfg.min_agreement, n_present)
        if mag >= cfg.high_conviction_magnitude and agree_ok:
            return "high"
        if mag >= cfg.high_conviction_magnitude / 2 and agree_ok:
            return "moderate"
        return "low"


# ---------------------------------------------------------------------------
# Expiry-level positioning summary (max-pain, put/call, center of gravity)
# ---------------------------------------------------------------------------
@dataclass
class PositioningSummary:
    expiry: dt.date
    put_call_oi_ratio: Optional[float]
    total_call_oi: float
    total_put_oi: float
    oi_center_of_gravity: Optional[float]   # OI-weighted mean strike
    max_pain: Optional[float]
    total_premium_notional: Optional[float]  # sum OI*mid*mult
    multiplier: float
    top_conviction: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "expiry": self.expiry.isoformat(),
            "put_call_oi_ratio": _num(self.put_call_oi_ratio),
            "total_call_oi": _num(self.total_call_oi),
            "total_put_oi": _num(self.total_put_oi),
            "oi_center_of_gravity": _num(self.oi_center_of_gravity),
            "max_pain": _num(self.max_pain),
            "total_premium_notional": _num(self.total_premium_notional),
            "multiplier": self.multiplier,
            "top_conviction": self.top_conviction,
        }


def summarize_positioning(rows: list[SnapshotRow], expiry: dt.date,
                          convictions: list[StrikeConviction],
                          multiplier: float = 100.0) -> PositioningSummary:
    calls = [r for r in rows if r.call_put == "C"]
    puts = [r for r in rows if r.call_put == "P"]
    tot_c = float(np.nansum([_g(r.open_interest) for r in calls])) if calls else 0.0
    tot_p = float(np.nansum([_g(r.open_interest) for r in puts])) if puts else 0.0
    pcr = (tot_p / tot_c) if tot_c > 0 else None

    # OI-weighted center of gravity
    ks = np.array([r.strike for r in rows], float)
    oi = np.array([_g(r.open_interest) for r in rows], float)
    m = np.isfinite(ks) & np.isfinite(oi) & (oi > 0)
    cog = float(np.sum(ks[m] * oi[m]) / np.sum(oi[m])) if m.any() else None

    # premium notional (capital at risk) = sum OI * mid * multiplier
    prem = 0.0
    have_prem = False
    for r in rows:
        if r.open_interest and r.mid_price and np.isfinite(r.open_interest) \
                and np.isfinite(r.mid_price):
            prem += float(r.open_interest) * float(r.mid_price) * multiplier
            have_prem = True

    max_pain = _max_pain(rows, multiplier)

    top = sorted(
        [c for c in convictions if c.composite in ("high", "conflicting")],
        key=lambda c: abs(c.magnitude) if c.magnitude else 0.0, reverse=True)[:10]

    return PositioningSummary(
        expiry=expiry, put_call_oi_ratio=pcr,
        total_call_oi=tot_c, total_put_oi=tot_p,
        oi_center_of_gravity=cog, max_pain=max_pain,
        total_premium_notional=prem if have_prem else None,
        multiplier=multiplier,
        top_conviction=[c.as_dict() for c in top])


def _max_pain(rows: list[SnapshotRow], multiplier: float) -> Optional[float]:
    """Strike minimizing total option-holder payoff (classic max-pain)."""
    strikes = sorted({r.strike for r in rows if np.isfinite(r.strike)})
    if not strikes:
        return None
    calls = [(r.strike, _g(r.open_interest)) for r in rows if r.call_put == "C"]
    puts = [(r.strike, _g(r.open_interest)) for r in rows if r.call_put == "P"]
    best_k, best_pain = None, None
    for s in strikes:
        pain = 0.0
        for k, o in calls:
            if np.isfinite(o) and s > k:
                pain += (s - k) * o
        for k, o in puts:
            if np.isfinite(o) and s < k:
                pain += (k - s) * o
        if best_pain is None or pain < best_pain:
            best_pain, best_k = pain, s
    return best_k


# ---------------------------------------------------------------------------
def _delta(cur, prev):
    if cur is None or prev is None:
        return None
    try:
        cf, pf = float(cur), float(prev)
        if cf != cf or pf != pf:
            return None
        return cf - pf
    except (TypeError, ValueError):
        return None


def _safe_ratio(a, b):
    try:
        if a is None or b is None:
            return None
        af, bf = float(a), float(b)
        if af != af or bf != bf or bf <= 0:
            return None
        return af / bf
    except (TypeError, ValueError):
        return None


def _g(x):
    try:
        return float(x) if x is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _intensity_score(level, series: np.ndarray, min_points: int) -> Optional[float]:
    """Score volume/OI turnover as ELEVATION ABOVE A QUIET BASELINE.

    Unlike a plain z-score, this does NOT de-mean against the series' own
    average -- a persistently high turnover ratio *is* the signal and must not
    be normalized away. We anchor 'quiet' at the low end of the strike's own
    recent turnover (a low percentile) and express today's level as multiples
    of the typical spread above that quiet floor. High and rising -> large
    positive; sitting at its usual quiet level -> ~0.
    """
    if level is None or not np.isfinite(level):
        return None
    s = series[np.isfinite(series)]
    if s.size < min_points:
        return None
    quiet = float(np.percentile(s, 20))          # baseline low-turnover level
    spread = float(np.percentile(s, 80) - quiet)  # typical elevation span
    if not np.isfinite(spread) or spread <= 0:
        # near-constant turnover: fall back to absolute convention
        # (vol/OI ~1 means one full turnover today -> strong).
        return float(min(level / 0.5, 4.0))
    return float((level - quiet) / spread)
