"""
Assemble a provider-agnostic OptionChain.

For the mock provider this simply delegates to build_chain(). For the live
BloombergProvider it orchestrates BDS (chain members) -> BDP (per-option vols,
strikes, expiries) -> group into ExpirySlices and compute forwards.

Kept separate from bloomberg.py so the parsing / grouping logic is unit-testable
without a Terminal.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections import defaultdict

import numpy as np

from .bloomberg import (ASSET_DEFAULTS, BloombergProvider, MockProvider,
                        OptionChain, OptionQuote, ExpirySlice, act365,
                        is_rate_future)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black76_price(F: float, K: float, T: float, sigma: float,
                   is_call: bool, shift: float = 0.0) -> float:
    """Undiscounted Black-76 price (r=0; we only need it to back out IV, and the
    discount factor cancels in the root-find)."""
    F = F + shift
    K = K + shift
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return F * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - F * _norm_cdf(-d1)


def implied_vol_from_price(price: float, F: float, K: float, T: float,
                           is_call: bool, shift: float = 0.0) -> float | None:
    """Back out Black-76 implied vol from an option mid price via bisection.
    Returns None if the price is outside no-arbitrage bounds or won't converge.
    """
    if not (price and price > 0) or F <= 0 or K <= 0 or T <= 0:
        return None
    # Intrinsic (forward, undiscounted) lower bound sanity check.
    Fs, Ks = F + shift, K + shift
    intrinsic = max(Fs - Ks, 0.0) if is_call else max(Ks - Fs, 0.0)
    if price < intrinsic - 1e-6:
        return None
    lo, hi = 1e-4, 5.0
    p_lo = _black76_price(F, K, T, lo, is_call, shift)
    p_hi = _black76_price(F, K, T, hi, is_call, shift)
    if math.isnan(p_lo) or math.isnan(p_hi):
        return None
    if not (p_lo <= price <= p_hi):
        # price above max vol we allow, or below min -> clamp attempt fails
        if price > p_hi:
            return None
        if price < p_lo:
            return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pm = _black76_price(F, K, T, mid, is_call, shift)
        if math.isnan(pm):
            return None
        if abs(pm - price) < 1e-6:
            return mid
        if pm < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Minimum time value (in price points) for a quote to carry vol information.
# SOFR/FF options tick in 0.0025-0.005, so anything at or inside intrinsic plus
# a tick is noise, not a price.
_MIN_TIME_VALUE = 0.005

# Sanity bounds on a rate-future PRICE vol (decimal). A 96 price moving 0.2-3%
# a year spans roughly 20-300bp of normal vol, which brackets any real market.
_RATE_PX_VOL_MIN = 0.0005
_RATE_PX_VOL_MAX = 0.10


def classify_asset(ticker: str) -> str:
    u = ticker.upper()
    # Rate futures must be tested BEFORE the .endswith("COMDTY") branch, which
    # would otherwise claim them.
    if is_rate_future(u):
        return "RATES_PRICE"
    if any(t in u for t in ("FED", "SOFR", " OIS", "RATE")):
        return "RATES"
    if u.endswith("INDEX"):
        return "EQ_INDEX"
    if u.endswith("EQUITY"):
        return "EQUITY"
    if u.endswith("CURNCY"):
        return "FX"
    if u.endswith("COMDTY"):
        return "CMDTY"
    return "EQUITY"


def build_chain(provider, underlying: str, n_expiries: int = 6,
                max_options: int = 1500,
                target_date: dt.date | None = None) -> OptionChain:
    """Dispatch to the correct builder based on provider type."""
    if isinstance(provider, MockProvider):
        return provider.build_chain(underlying, n_expiries=n_expiries)
    if isinstance(provider, BloombergProvider):
        return _build_live(provider, underlying, n_expiries, max_options,
                           target_date=target_date)
    raise TypeError(f"Unknown provider type: {type(provider)}")


def _select_members_by_expiry(members: list[str], as_of: dt.date,
                              n_expiries: int, max_options: int,
                              target: dt.date | None = None) -> list[str]:
    """Group chain members by their (ticker-parsed) expiry and keep a spread
    across the term structure -- NOT just the nearest N members.

    Ensures the expiry closest to `target` is included, then adds a spread of
    other expiries, capping total members at `max_options`.
    """
    by_exp: dict[dt.date, list[str]] = defaultdict(list)
    undated: list[str] = []
    for tk in members:
        e = _expiry_from_ticker(tk)
        if e is None or e <= as_of:
            undated.append(tk)
        else:
            by_exp[e].append(tk)

    if not by_exp:
        # Can't parse expiries from tickers -> fall back to original behaviour.
        return members[:max_options]

    all_exps = sorted(by_exp.keys())

    chosen_exps: list[dt.date] = []
    # 1) Always include the expiry nearest the target date (if given).
    if target is not None:
        nearest = min(all_exps, key=lambda e: abs((e - target).days))
        chosen_exps.append(nearest)
    # 2) Add a spread across the remaining term structure until we have
    #    n_expiries. Prefer monthly-ish spacing by walking evenly.
    remaining = [e for e in all_exps if e not in chosen_exps]
    if remaining and len(chosen_exps) < n_expiries:
        need = n_expiries - len(chosen_exps)
        if need >= len(remaining):
            chosen_exps.extend(remaining)
        else:
            idxs = np.linspace(0, len(remaining) - 1, need).astype(int)
            chosen_exps.extend(remaining[i] for i in sorted(set(idxs)))
    chosen_exps = sorted(set(chosen_exps))

    # Collect members for the chosen expiries, capped at max_options.
    out: list[str] = []
    per_exp_cap = max(20, max_options // max(len(chosen_exps), 1))
    for e in chosen_exps:
        out.extend(by_exp[e][:per_exp_cap])
    return out[:max_options]


def _build_live(bbg: BloombergProvider, underlying: str, n_expiries: int,
                max_options: int, target_date: dt.date | None = None) -> OptionChain:
    """Assemble an OptionChain from a live Bloomberg connection.

    Robust to the usual xbbg quirks: mixed column names, missing IVOL, and
    string expiry/strike fields. Any option row lacking a positive implied vol
    is dropped.
    """
    asset_class = classify_asset(underlying)
    defaults = ASSET_DEFAULTS[asset_class]
    as_of = dt.date.today()

    spot = bbg.spot(underlying)

    # Chain members. Instead of blindly taking the first max_options (which are
    # typically all near-dated weeklies for SPX), parse each ticker's expiry and
    # keep a spread across the term structure, guaranteeing the expiry nearest
    # the requested target date is included.
    members = bbg.chain_tickers(underlying, call_put="C")
    members = _select_members_by_expiry(members, as_of, n_expiries,
                                        max_options, target=target_date)

    df = bbg.option_fields(members)
    # df is indexed by ticker with columns per field (xbbg lowercases fields).
    cols = {c.lower(): c for c in df.columns}

    def col(name, *alts):
        for n in (name, *alts):
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_iv = col("ivol_mid", "ivol", "3mo_call_imp_vol")
    c_k = col("opt_strike_px", "strike")
    c_exp = col("opt_expire_dt", "expiry")
    c_pc = col("opt_put_call", "put_call")
    c_undl = col("opt_undl_px")
    c_bid, c_ask, c_last = col("px_bid"), col("px_ask"), col("px_last")
    c_oi, c_vol = col("open_int"), col("px_volume")

    by_exp: dict[dt.date, list[OptionQuote]] = defaultdict(list)
    undl_by_exp: dict[dt.date, list[float]] = defaultdict(list)

    # Rate futures quote IVOL_MID on the RATE, not on the price we fit, so the
    # vol must come from the mid price instead. See the block below.
    prefer_price_iv = (asset_class == "RATES_PRICE")

    # Track how options are dropped so the failure is explainable.
    n_rows = 0
    n_no_strike = n_no_expiry = n_no_iv = n_kept = 0
    n_no_tv = n_bad_iv = 0

    for tk, row in df.iterrows():
        n_rows += 1
        try:
            # ---- strike (field, else parse from ticker) ----
            K = _f(row.get(c_k)) if c_k else None
            if K is None or K <= 0:
                K = _strike_from_ticker(str(tk))
            if K is None or K <= 0:
                n_no_strike += 1
                continue
            # ---- expiry (field, else parse from ticker) ----
            exp = _to_date(row.get(c_exp)) if c_exp else None
            if exp is None:
                exp = _expiry_from_ticker(str(tk))
            if exp is None or exp <= as_of:
                n_no_expiry += 1
                continue
            T = act365(as_of, exp)
            # put/call: field, else parse from ticker, else moneyness.
            pc_raw = str(row.get(c_pc)).strip().upper()[:1] if c_pc else ""
            if pc_raw not in ("C", "P", "1", "0"):
                parsed = _parse_opt_ticker(str(tk))
                pc_raw = parsed["call_put"] if parsed else ("C" if K >= spot else "P")
            pc = "C" if pc_raw in ("C", "1") else "P"

            # ---- prices / mid ----
            bid = _f(row.get(c_bid)) if c_bid else None
            ask = _f(row.get(c_ask)) if c_ask else None
            last = _f(row.get(c_last)) if c_last else None
            mid = None
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                mid = 0.5 * (bid + ask)
            elif last is not None and last > 0:
                mid = last

            # ---- implied vol ----
            # Equity/FX/commodity: trust IVOL_MID, which is quoted on the thing
            # we fit, and convert percent -> decimal.
            #
            # RATE FUTURES: do NOT. Confirmed live via diagnose_rate_vol.py that
            # IVOL_MID here is a lognormal vol on the RATE, quoted in percent --
            # median IVOL_MID / (vol implied by the mid price) came out at
            # ~2475, i.e. (price/rate) x 100 = 24.2 x 100. Bloomberg returns
            # ~20, the app divided to 0.20, and 0.20 is 20% of the 3.97 RATE
            # (~79bp normal), not 20% of the 96.03 PRICE. Used as a price vol it
            # overstates the density ~24x: the RND then spanned -5% to +12% with
            # a fit RMSE of 31.6 vol points and rho pinned at -0.999.
            #
            # The mid PRICE is convention-free, so back the vol out of it.
            # Note the `iv > 3.0` percent heuristic is ALSO unsafe here for a
            # second reason: rate-future price vols are ~1%, below the threshold,
            # so a genuine percent would sail through unconverted.
            iv = None
            if not prefer_price_iv:
                iv = _f(row.get(c_iv)) if c_iv else None
                if iv is not None and iv > 0:
                    if iv > 3.0:            # percent -> decimal
                        iv = iv / 100.0
                else:
                    iv = None
            if iv is None and mid is not None:
                F_est = spot  # refined per-expiry later; adequate for IV inversion
                sh = float(defaults["shift"])
                # An option with no meaningful time value carries no volatility
                # information: the back-out collapses onto the solver's floor and
                # produces the flat junk line seen at deep strikes, which then
                # drags the whole calibration. Reject rather than fit.
                Fs, Ks = F_est + sh, K + sh
                intrinsic = max(Fs - Ks, 0.0) if pc == "C" else max(Ks - Fs, 0.0)
                if (mid - intrinsic) < _MIN_TIME_VALUE:
                    n_no_tv += 1
                    continue
                iv = implied_vol_from_price(mid, F_est, K, T,
                                            is_call=(pc == "C"), shift=sh)
            if iv is None or not (iv > 0):
                n_no_iv += 1
                continue
            if prefer_price_iv and not (_RATE_PX_VOL_MIN <= iv <= _RATE_PX_VOL_MAX):
                # Sanity bound on a rate-future PRICE vol. Plausible is ~0.2-3%;
                # anything outside this is a bad quote, not a market view.
                n_bad_iv += 1
                continue

            n_kept += 1
            q = OptionQuote(
                strike=K, expiry=exp, call_put=pc, implied_vol=iv,
                bid=bid,
                ask=ask,
                mid_price=mid,
                open_interest=_f(row.get(c_oi)) if c_oi else None,
                volume=_f(row.get(c_vol)) if c_vol else None,
            )
            by_exp[exp].append(q)
            if c_undl:
                u = _f(row.get(c_undl))
                if u:
                    undl_by_exp[exp].append(u)
        except Exception:
            continue

    # Leave a breadcrumb: if a chain comes back thin, the reason should be
    # readable rather than requiring a debugger.
    import logging
    logging.getLogger(__name__).info(
        "build_chain %s [%s]: %d rows -> %d kept "
        "(dropped: no_strike=%d no_expiry=%d no_time_value=%d no_iv=%d "
        "implausible_iv=%d) vol_source=%s",
        underlying, asset_class, n_rows, n_kept, n_no_strike, n_no_expiry,
        n_no_tv, n_no_iv, n_bad_iv,
        "mid_price_backout" if prefer_price_iv else "IVOL_MID",
    )
    if n_kept == 0:
        raise ValueError(
            f"No usable option quotes for {underlying!r} out of {n_rows} chain "
            f"rows (no_strike={n_no_strike}, no_expiry={n_no_expiry}, "
            f"no_time_value={n_no_tv}, no_iv={n_no_iv}, "
            f"implausible_iv={n_bad_iv}). "
            + ("Rate futures take their vol from the mid price, so missing "
               "bid/ask is the usual cause." if prefer_price_iv else "")
        )

    # Keep the nearest n_expiries with a reasonable number of strikes.
    exps = sorted(e for e, qs in by_exp.items() if len(qs) >= 5)[:n_expiries]
    slices = []
    for e in exps:
        qs = by_exp[e]
        F = float(np.median(undl_by_exp[e])) if undl_by_exp.get(e) else spot
        slices.append(ExpirySlice(expiry=e, forward=F, T=act365(as_of, e), quotes=qs))

    return OptionChain(underlying=underlying, asset_class=asset_class, spot=float(spot),
                       as_of=as_of, expiries=slices, source="bloomberg",
                       shift=float(defaults["shift"]))


# Listed-option ticker parsing, e.g.
#   'SPXW US 07/17/26 C4300 Index'  -> expiry 2026-07-17, call, strike 4300
#   'SPX US 12/19/25 P5000 Index'   -> expiry 2025-12-19, put,  strike 5000
import re as _re
_OPT_RE = _re.compile(
    r"(?P<mm>\d{1,2})/(?P<dd>\d{1,2})/(?P<yy>\d{2,4})\s+"
    r"(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)", _re.IGNORECASE)


def _parse_opt_ticker(tk: str):
    m = _OPT_RE.search(tk or "")
    if not m:
        return None
    yy = int(m.group("yy"))
    if yy < 100:
        yy += 2000
    try:
        exp = dt.date(yy, int(m.group("mm")), int(m.group("dd")))
    except Exception:
        exp = None
    return {"expiry": exp,
            "call_put": m.group("cp").upper(),
            "strike": float(m.group("strike"))}


def _expiry_from_ticker(tk: str):
    p = _parse_opt_ticker(tk)
    return p["expiry"] if p else None


def _strike_from_ticker(tk: str):
    p = _parse_opt_ticker(tk)
    return p["strike"] if p else None


def _to_date(v):
    if isinstance(v, dt.date):
        return v
    if isinstance(v, dt.datetime):
        return v.date()
    try:
        import pandas as pd
        return pd.to_datetime(v).date()
    except Exception:
        return None


def _f(v):
    try:
        if v is None:
            return None
        f = float(v)
        return None if (f != f) else f
    except Exception:
        return None


def _meta_from_bdp(bbg, tickers: list[str]) -> dict:
    """Resolve (expiry, strike, call_put) via BDP for tickers whose metadata is
    not recoverable from the ticker string.

    Needed for FUTURES options. Equity/index tickers embed the date
    ('SPXW US 07/17/26 C4300 Index'), but an IMM futures option does not:
    'SFRZ6C 96.00 Comdty' carries the strike and C/P yet has NO expiry in the
    string at all. No regex can recover it -- it has to come from Bloomberg.
    """
    out: dict = {}
    if not tickers:
        return out
    try:
        df = bbg.option_fields(tickers)
    except Exception:
        return out
    if df is None or getattr(df, "empty", True):
        return out

    cols = {str(c).lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_k = col("opt_strike_px", "strike")
    c_exp = col("opt_expire_dt", "expiry")
    c_pc = col("opt_put_call", "put_call")

    for tk, row in df.iterrows():
        K = _f(row.get(c_k)) if c_k else None
        exp = _to_date(row.get(c_exp)) if c_exp else None
        if K is None or K <= 0 or exp is None:
            continue
        pc_raw = str(row.get(c_pc)).strip().upper()[:1] if c_pc else ""
        if pc_raw not in ("C", "P", "1", "0"):
            # Futures options encode C/P in the ticker root, e.g. SFRZ6C / SFRZ6P.
            m = re.search(r"([CP])\s*\d", str(tk).upper())
            pc_raw = m.group(1) if m else "C"
        out[tk] = {"expiry": exp, "strike": float(K),
                   "call_put": "C" if pc_raw in ("C", "1") else "P"}
    return out


def backfill_history_rows(bbg: BloombergProvider, underlying: str,
                          days: int = 90, n_expiries: int = 6,
                          max_options: int = 800):
    """Seed the local positioning store from Bloomberg bdh history.

    For the current chain members (a spread across expiries), pull daily
    OPEN_INT / IVOL_MID / PX_LAST history and emit SnapshotRow objects, one per
    (date, strike, call_put). Best-effort: Bloomberg's granular OI/volume
    history is patchy, so whatever bdh returns is stored and gaps are simply
    absent. Returns a list of SnapshotRow (imported lazily to avoid a cycle).
    """
    from .positioning_store import SnapshotRow

    as_of = dt.date.today()
    start = as_of - dt.timedelta(days=days)

    members = bbg.chain_tickers(underlying, call_put="C")
    members = _select_members_by_expiry(members, as_of, n_expiries,
                                         max_options, target=None)
    if not members:
        return []

    # Map each ticker to its (strike, expiry, call_put) once. Ticker parsing is
    # the fast path and works for equity/index conventions; futures options
    # ('SFRZ6C 96.00 Comdty') carry no expiry in the string, so anything that
    # does not parse is resolved via BDP instead.
    meta = {}
    unparsed = []
    for tk in members:
        p = _parse_opt_ticker(str(tk))
        if p and p.get("expiry") and p.get("strike"):
            meta[tk] = p
        else:
            unparsed.append(tk)
    if unparsed:
        meta.update(_meta_from_bdp(bbg, unparsed))

    if not meta:
        # Without this guard bdh() is called with an empty securities list and
        # blpapi raises the opaque "securities is required for
        # HistoricalDataRequest". Fail with something actionable instead.
        raise ValueError(
            f"Could not resolve strike/expiry for any of {len(members)} chain "
            f"members of {underlying!r} (tried ticker parsing then BDP "
            f"OPT_STRIKE_PX/OPT_EXPIRE_DT). Sample: {list(members)[:3]}"
        )

    flds = ["OPEN_INT", "IVOL_MID", "PX_LAST", "PX_VOLUME"]
    long = bbg.bdh_fields(list(meta.keys()), flds, start, as_of)
    if long is None or getattr(long, "empty", True):
        return []

    import pandas as pd
    # Pivot to (date, ticker) x field so we can assemble rows.
    piv = long.pivot_table(index=["date", "ticker"], columns="field",
                           values="value", aggfunc="last")
    piv.columns = [str(c).lower() for c in piv.columns]

    rows = []
    for (d, tk), rec in piv.iterrows():
        p = meta.get(tk)
        if not p:
            continue
        dd = d.date() if hasattr(d, "date") else d
        rows.append(SnapshotRow(
            as_of=dd, underlying=underlying,
            expiry=p["expiry"], strike=float(p["strike"]),
            call_put=p.get("call_put") or "C",
            open_interest=_f(rec.get("open_int")),
            volume=_f(rec.get("px_volume")),
            implied_vol=_f(rec.get("ivol_mid")),
            mid_price=_f(rec.get("px_last")),
        ))
    return rows
