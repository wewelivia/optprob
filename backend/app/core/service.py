"""
Orchestration service: ties data -> SABR -> Breeden-Litzenberger -> event
probability into single callable used by the API layer.
"""
from __future__ import annotations

import datetime as dt
import math
from functools import lru_cache

import numpy as np

from ..data.bloomberg import (ASSET_DEFAULTS, RATE_FUTURE_REF, BloombergProvider,
                              MockProvider, get_provider)
from ..data.chain_builder import build_chain, classify_asset
from ..data.positioning_store import PositioningStore, rows_from_chain
from .sabr import calibrate_sabr
from .breeden_litzenberger import extract_rnd, to_rate_space
from .event_parser import parse_condition, compute_probability
from .conviction import (ConvictionEngine, ConvictionConfig,
                         summarize_positioning)

# Contract multiplier by asset class (for premium/notional).
_MULTIPLIER = {"EQ_INDEX": 100.0, "EQUITY": 100.0, "FX": 1.0,
               "CMDTY": 1.0, "RATES": 1.0,
               # 3M SOFR convention: $25 per basis point => $2,500 per price
               # point. NOTE this is contract-specific -- 30-day fed funds (FF)
               # is ~$41.67/bp on a 1M notional, Euribor is EUR 25/bp. Premium
               # notional for non-SOFR rate futures will be wrong until this is
               # keyed by root rather than asset class.
               "RATES_PRICE": 2500.0}

_STORE = PositioningStore()


# Cache chains briefly so repeated requests on the same underlying are fast.
_CHAIN_CACHE: dict[str, tuple[float, object]] = {}
_CHAIN_TTL = 60.0  # seconds


def _get_chain(underlying: str, prefer_live: bool = True,
               target_date=None):
    import time
    now = time.time()
    key = f"{underlying}|{target_date.isoformat() if target_date else '-'}"
    hit = _CHAIN_CACHE.get(key)
    if hit and (now - hit[0] < _CHAIN_TTL):
        return hit[1]
    provider = get_provider(prefer_live=prefer_live)
    chain = build_chain(provider, underlying, target_date=target_date)
    _CHAIN_CACHE[key] = (now, chain)
    # Persist a strike-level snapshot for the conviction/positioning history.
    # Only for live chains (mock data would pollute the real series).
    try:
        if getattr(chain, "source", "mock") != "mock":
            _STORE.write_snapshot(rows_from_chain(chain))
    except Exception:  # persistence must never break the main pipeline
        pass
    return chain


def _shape(obj) -> dict | str:
    """Rich description of an xbbg return: shape, column labels (including
    MultiIndex tuples), row index, and a small preview of the actual cells so
    we can tell an empty frame from an oddly-shaped one."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": list(obj.shape),
                "empty": bool(obj.empty),
                "columns_type": type(obj.columns).__name__,
                "columns": [list(c) if isinstance(c, tuple) else c
                            for c in list(obj.columns)[:12]],
                "index": [str(i) for i in list(obj.index)[:6]],
                "preview": obj.head(3).to_dict(orient="list") if not obj.empty else {},
            }
        if isinstance(obj, pd.Series):
            return {"type": "Series", "len": len(obj),
                    "index": [str(i) for i in list(obj.index)[:12]],
                    "preview": obj.head(6).tolist()}
    except Exception as e:
        return f"{type(obj).__name__} (shape-introspection error: {e})"
    return f"{type(obj).__name__}"


def diagnose(underlying: str) -> dict:
    """Probe the live Bloomberg path one call at a time, capturing the raw
    return shape and any error per step. Never raises -- returns a report the
    UI/terminal can show so we can pinpoint exactly where a chain build fails.
    """
    report: dict = {"underlying": underlying,
                    "classified_as": classify_asset(underlying),
                    "steps": []}

    prov = get_provider(prefer_live=True)
    is_live = isinstance(prov, BloombergProvider)
    report["provider"] = "bloomberg" if is_live else "mock"
    if not is_live:
        report["note"] = ("No live Terminal detected (xbbg/blpapi not importable or "
                          "not connected). The app is using synthetic surfaces.")
        # Still show the mock builds cleanly.
        try:
            chain = build_chain(prov, underlying)
            report["steps"].append({"step": "mock_build", "ok": True,
                                    "expiries": len(chain.expiries),
                                    "asset_class": chain.asset_class})
        except Exception as e:
            report["steps"].append({"step": "mock_build", "ok": False, "error": repr(e)})
        return report

    # ---- live probes ----
    def probe(name, fn):
        import traceback as _tb
        entry = {"step": name}
        try:
            out = fn()
            entry["ok"] = True
            entry["result"] = out
        except Exception as e:
            entry["ok"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = _tb.format_exc().splitlines()[-4:]
        report["steps"].append(entry)
        return entry.get("result")

    # 1) raw spot frame shape
    def _raw_spot():
        df = prov._xbbg.bdp(tickers=underlying, flds=["PX_LAST"])
        return _shape(df)
    probe("bdp_px_last_shape", _raw_spot)

    # 2) parsed spot
    spot = probe("spot_value", lambda: prov.spot(underlying))

    # 3) raw OPT_CHAIN shape
    def _raw_chain():
        df = prov._xbbg.bds(tickers=underlying, flds="OPT_CHAIN")
        return _shape(df)
    probe("bds_opt_chain_shape", _raw_chain)

    # 4) parsed chain members (count + sample)
    def _members():
        m = prov.chain_tickers(underlying)
        return {"count": len(m), "sample": m[:5]}
    members_res = probe("chain_members", _members)

    # 5) option fields for a small sample -- show actual values so we can see
    #    whether IV / strike / expiry are populated (the usual cause of
    #    "no expiries available").
    def _fields():
        m = prov.chain_tickers(underlying)[:25]
        df = prov.option_fields(m)
        cols = list(getattr(df, "columns", []))
        # populated-count per column
        populated = {}
        try:
            for c in cols:
                nonnull = int(df[c].notna().sum())
                populated[str(c)] = nonnull
        except Exception:
            pass
        # first few full rows (ticker + all fields)
        rows = []
        try:
            for tk, row in df.head(5).iterrows():
                rows.append({"ticker": str(tk),
                             **{str(c): str(row[c]) for c in cols}})
        except Exception as e:
            rows = [f"row-dump error: {e}"]
        return {"sample_tickers_queried": m[:5],
                "shape": _shape(df),
                "columns": [str(c) for c in cols],
                "populated_per_column": populated,
                "sample_rows": rows}
    probe("option_fields_sample", _fields)

    # 5b) expiry grouping breakdown: how many strikes land on each expiry
    def _expiry_breakdown():
        from ..data.chain_builder import _to_date
        m = prov.chain_tickers(underlying)
        df = prov.option_fields(m[:400])  # bounded sample
        cols = {str(c).lower(): c for c in df.columns}
        def col(name, *alts):
            for n in (name, *alts):
                if n.lower() in cols:
                    return cols[n.lower()]
            return None
        c_iv = col("ivol_mid", "ivol")
        c_k = col("opt_strike_px", "strike")
        c_exp = col("opt_expire_dt", "expiry")
        counts = {}
        iv_ok = k_ok = exp_ok = 0
        for _, row in df.iterrows():
            try:
                iv = row[c_iv] if c_iv else None
                k = row[c_k] if c_k else None
                e = _to_date(row[c_exp]) if c_exp else None
                if iv is not None and str(iv).strip() not in ("", "nan", "None"):
                    iv_ok += 1
                if k is not None and str(k).strip() not in ("", "nan", "None"):
                    k_ok += 1
                if e is not None:
                    exp_ok += 1
                    counts[e.isoformat()] = counts.get(e.isoformat(), 0) + 1
            except Exception:
                continue
        top = dict(sorted(counts.items(), key=lambda x: -x[1])[:10])
        return {"resolved_cols": {"iv": str(c_iv), "strike": str(c_k), "expiry": str(c_exp)},
                "rows_scanned": int(min(len(m), 400)),
                "iv_populated": iv_ok, "strike_populated": k_ok,
                "expiry_parsed": exp_ok,
                "strikes_per_expiry_top10": top,
                "expiries_with_5plus": sum(1 for v in counts.values() if v >= 5)}
    probe("expiry_breakdown", _expiry_breakdown)

    # 6) full build
    def _build():
        chain = build_chain(prov, underlying)
        return {"expiries": len(chain.expiries),
                "asset_class": chain.asset_class,
                "first_expiry": chain.expiries[0].expiry.isoformat() if chain.expiries else None,
                "n_strikes_first": (len({q.strike for q in chain.expiries[0].quotes})
                                    if chain.expiries else 0)}
    probe("full_build_chain", _build)

    return report


def get_chain_info(underlying: str, prefer_live: bool = True) -> dict:
    chain = _get_chain(underlying, prefer_live)
    return {
        "underlying": chain.underlying,
        "asset_class": chain.asset_class,
        "spot": chain.spot,
        "as_of": chain.as_of.isoformat(),
        "source": chain.source,
        "shift": chain.shift,
        "expiries": [
            {"expiry": e.expiry.isoformat(), "T": e.T, "forward": e.forward,
             "n_strikes": len({q.strike for q in e.quotes})}
            for e in chain.expiries
        ],
    }


def _grid_bounds(chain, sl, strikes):
    if chain.asset_class == "RATES":
        span = max(4.0, 6.0 * sl.forward * math.sqrt(max(sl.T, 0.05)) / 4.0)
        return sl.forward - min(sl.forward + chain.shift - 1e-6, span), sl.forward + span
    if chain.asset_class == "RATES_PRICE":
        # Price space, bounded near 100. The generic branch below would give
        # 0.4*min .. 1.7*max, i.e. ~38 to ~165 for strikes around 96 -- absurd
        # for a (100 - rate) price and it would waste most of the grid. Pad the
        # listed strikes instead. Cap at 110 rather than 100 so negative rates
        # remain representable.
        lo = float(strikes.min()) - 3.0
        hi = float(strikes.max()) + 3.0
        return max(lo, 0.0), min(hi, 110.0)
    return float(strikes.min() * 0.4), float(strikes.max() * 1.7)


def _fmt_odds(p: float) -> str:
    if p <= 0:
        return "~0 (effectively impossible)"
    if p >= 1:
        return "~1 (effectively certain)"
    # implied "X to 1 against"
    against = (1 - p) / p
    if against >= 1:
        return f"{against:.1f} to 1 against"
    return f"{1/against:.1f} to 1 on"


def compute_positioning(underlying: str, condition: str | None = None,
                        expiry: str | None = None,
                        prefer_live: bool = True,
                        short_window: int = 5,
                        trend_window: int = 20,
                        context_window: int = 60) -> dict:
    """Conviction / positioning read for the expiry matching the target date.

    Uses the locally-accumulated strike-level snapshot history (written on each
    live chain build) to compute per-strike conviction, plus expiry-level
    positioning stats (put/call OI, center of gravity, max-pain, premium
    notional). Deltas need >=2 stored snapshots on different dates; a one-time
    bdh backfill can seed this (see /api/backfill).
    """
    # Determine target expiry the same way the distribution does.
    target_date = None
    if condition:
        try:
            target_date = parse_condition(condition).target_date
        except Exception:
            target_date = None

    chain = _get_chain(underlying, prefer_live, target_date=target_date)
    if not chain.expiries:
        raise ValueError(f"No option expiries available for {underlying!r}")

    if expiry:
        target = dt.date.fromisoformat(expiry)
        sl = min(chain.expiries, key=lambda e: abs((e.expiry - target).days))
    elif target_date is not None:
        sl = chain.nearest_expiry(target_date)
    else:
        sl = chain.expiries[0]

    cfg = ConvictionConfig(short_window=short_window,
                           trend_window=trend_window,
                           context_window=context_window)
    eng = ConvictionEngine(_STORE, cfg)
    convictions = eng.compute_for_expiry(chain.underlying, sl.expiry,
                                         as_of=chain.as_of)

    mult = _MULTIPLIER.get(chain.asset_class, 100.0)
    rows_today = _STORE.snapshot_on(chain.underlying, chain.as_of,
                                    expiry=sl.expiry)
    if not rows_today:
        # store not yet populated for today -> synthesize from the live chain
        rows_today = [r for r in rows_from_chain(chain) if r.expiry == sl.expiry]
    summary = summarize_positioning(rows_today, sl.expiry, convictions, mult)

    # --- rate-space display mapping ----------------------------------------
    # The store stays in PRICE space, which is the contract's real strike and
    # what Bloomberg returns; only the display is mapped, so the OI chart lines
    # up with the rate-space PDF the frontend overlays on it. Without this the
    # two axes silently disagree.
    #
    # call_put is deliberately NOT relabelled: a call on the price is a put on
    # the rate, and quietly flipping it would misrepresent the actual contract.
    # It stays as the real contract type; `rate_space` tells the reader to
    # interpret it accordingly.
    rate_space = chain.asset_class == "RATES_PRICE"
    strikes_out = [c.as_dict() for c in convictions]
    summary_out = summary.as_dict()
    forward_out = sl.forward
    spot_out = chain.spot
    if rate_space:
        for d in strikes_out:
            if d.get("strike") is not None:
                d["strike_price_space"] = float(d["strike"])
                d["strike"] = float(RATE_FUTURE_REF - d["strike"])
        strikes_out.sort(key=lambda d: (d["strike"], d.get("call_put") or ""))
        for k in ("oi_center_of_gravity", "max_pain"):
            v = summary_out.get(k)
            if v is not None:
                summary_out[f"{k}_price_space"] = float(v)
                summary_out[k] = float(RATE_FUTURE_REF - v)
        for d in summary_out.get("top_conviction") or []:
            if isinstance(d, dict) and d.get("strike") is not None:
                d["strike_price_space"] = float(d["strike"])
                d["strike"] = float(RATE_FUTURE_REF - d["strike"])
        forward_out = float(RATE_FUTURE_REF - sl.forward)
        spot_out = (float(RATE_FUTURE_REF - chain.spot)
                    if chain.spot is not None else None)

    n_dates = len(_STORE.available_dates(chain.underlying, before=chain.as_of))
    return {
        "underlying": chain.underlying,
        "asset_class": chain.asset_class,
        "source": chain.source,
        "expiry": sl.expiry.isoformat(),
        "forward": forward_out,
        "spot": spot_out,
        "rate_space": rate_space,
        "rate_future_ref": RATE_FUTURE_REF if rate_space else None,
        "as_of": chain.as_of.isoformat(),
        "history_days": n_dates,
        "deltas_available": n_dates >= 2,
        "windows": {"short": short_window, "trend": trend_window,
                    "context": context_window},
        "weights": {"iv": cfg.w_iv, "vol_oi": cfg.w_voloi, "oi": cfg.w_oi},
        "summary": summary_out,
        "strikes": strikes_out,
    }


def backfill_positioning(underlying: str, days: int = 90,
                         prefer_live: bool = True) -> dict:
    """One-time seed of the local snapshot store from Bloomberg bdh history so
    deltas work without waiting for days of live accumulation.

    Pulls per-contract OPEN_INT / IVOL_MID / PX history for the current chain
    members and writes one snapshot row per (date, strike, cp). Bloomberg's
    granular OI/volume history is patchy, so this is best-effort: whatever bdh
    returns is stored; gaps are simply absent from the series.
    """
    provider = get_provider(prefer_live=prefer_live)
    if isinstance(provider, MockProvider):
        return {"seeded": 0, "note": "mock provider; backfill skipped"}
    from ..data.chain_builder import backfill_history_rows
    rows = backfill_history_rows(provider, underlying, days=days)
    n = _STORE.write_snapshot(rows, replace=False)
    return {"underlying": underlying, "seeded": n,
            "days_requested": days}


def compute_distribution(underlying: str, condition: str,
                         beta: float | None = None, r: float = 0.0,
                         force_percent: bool | None = None,
                         expiry: str | None = None,
                         prefer_live: bool = True,
                         n_out: int = 250) -> dict:
    """Full pipeline: returns everything the frontend needs to render."""
    # Parse the condition first so we know the target date, then build the chain
    # around it (guarantees the expiry nearest the target is fetched).
    fp_pre = force_percent
    if fp_pre is None:
        fp_pre = classify_asset(underlying) in ("RATES", "RATES_PRICE")
    spec = parse_condition(condition, force_percent=fp_pre)

    chain = _get_chain(underlying, prefer_live, target_date=spec.target_date)
    if not chain.expiries:
        raise ValueError(f"No option expiries available for {underlying!r}")

    # Re-derive percent semantics now that we know the true asset class.
    fp = force_percent
    if fp is None:
        fp = chain.asset_class in ("RATES", "RATES_PRICE")
    if fp != fp_pre:
        spec = parse_condition(condition, force_percent=fp)

    # Choose expiry
    if expiry:
        target = dt.date.fromisoformat(expiry)
        sl = min(chain.expiries, key=lambda e: abs((e.expiry - target).days))
    else:
        sl = chain.nearest_expiry(spec.target_date)

    strikes, mvols = sl.smile()
    if len(strikes) < 3:
        raise ValueError(f"Too few strikes ({len(strikes)}) to calibrate at expiry {sl.expiry}")

    b = beta if beta is not None else ASSET_DEFAULTS[chain.asset_class]["beta"]
    params = calibrate_sabr(sl.forward, strikes, mvols, sl.T, beta=b, shift=chain.shift)

    lo, hi = _grid_bounds(chain, sl, strikes)
    rnd = extract_rnd(params, r=r, strike_lo=lo, strike_hi=hi, n_grid=1200)

    # --- rate-space transform for IMM-style rate futures --------------------
    # Everything above ran in PRICE space, which is where the options are
    # struck and where Bloomberg quotes the vols, so the SABR fit and the BL
    # second derivative are untouched by this. Only the finished density is
    # mapped, so the strategist can ask "above 4%" instead of translating to
    # "below 96" in their head.
    rate_space = chain.asset_class == "RATES_PRICE"
    forward_out = sl.forward
    if rate_space:
        rnd = to_rate_space(rnd, ref=RATE_FUTURE_REF)
        forward_out = float(RATE_FUTURE_REF - sl.forward)

    ev = compute_probability(rnd, spec)

    # Fitted vols at market strikes for the smile overlay. In rate space the
    # x-axis is mapped to rates so it lines up with the density; the vols
    # themselves stay PRICE-space Black-76 vols (there is no such thing as a
    # rate vol here without refitting), and both spaces are returned so the
    # frontend never has to guess.
    fitted = params.vol(strikes)
    if rate_space:
        smile = [{"strike": float(RATE_FUTURE_REF - k),
                  "strike_price_space": float(k),
                  "market_vol": float(mv), "fitted_vol": float(fv)}
                 for k, mv, fv in zip(strikes, mvols, fitted)]
        smile.sort(key=lambda d: d["strike"])
    else:
        smile = [{"strike": float(k), "strike_price_space": float(k),
                  "market_vol": float(mv), "fitted_vol": float(fv)}
                 for k, mv, fv in zip(strikes, mvols, fitted)]

    # Downsample RND arrays for transport
    idx = np.linspace(0, len(rnd.strikes) - 1, min(n_out, len(rnd.strikes))).astype(int)
    grid = rnd.strikes[idx].tolist()
    pdf = rnd.pdf[idx].tolist()
    cdf = rnd.cdf[idx].tolist()

    p = ev["probability"]
    return {
        "underlying": chain.underlying,
        "asset_class": chain.asset_class,
        "source": chain.source,
        "expiry": sl.expiry.isoformat(),
        "T": sl.T,
        "forward": forward_out,
        "forward_price_space": float(sl.forward) if rate_space else None,
        "rate_space": rate_space,
        "rate_future_ref": RATE_FUTURE_REF if rate_space else None,
        "is_percent": spec.is_percent,
        "sabr": params.as_dict(),
        "smile": smile,
        "grid": grid,
        "pdf": pdf,
        "cdf": cdf,
        "stats": rnd.stats(),
        "probability": p,
        "condition": ev["condition"],
        "direction": ev["direction"],
        "threshold": ev["threshold"],
        "threshold_hi": ev["threshold_hi"],
        "target_date": ev["target_date"],
        "complement": 1.0 - p,
        "odds": _fmt_odds(p),
    }
