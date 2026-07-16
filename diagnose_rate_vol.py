"""
Determine what IVOL_MID actually means for rate-future options.

Why this exists
---------------
chain_builder guesses units with:

    if iv > 3.0: iv = iv / 100.0     # "must be a percent"

That holds for equities (SPX ~16 -> 0.16). It BREAKS for rate futures, whose
PRICE vol is genuinely tiny: a 96.05 price barely moves even when the rate
swings, so Bloomberg may return ~1.2 meaning 1.2%, the check fails, and the app
silently fits 120% vol.

There is also a second, nastier ambiguity. Rates desks often quote NORMAL
(Bachelier) vols rather than lognormal. And because

    sigma_normal = sigma_lognormal * F,   with F ~ 96 ~ 100

a normal vol in price points and a lognormal vol in percent are numerically
almost identical here. You CANNOT tell them apart by looking at the number.

The way out: ignore IVOL_MID and back the lognormal vol out of the actual mid
PRICE. That is convention-free -- the price is the price. Comparing the two
tells us definitively what IVOL_MID is:

    raw / backed_out ~ 100  -> IVOL_MID is a lognormal PERCENT  (needs /100)
    raw / backed_out ~ 1    -> IVOL_MID is already a decimal    (fine as-is)
    raw / backed_out ~ 96   -> consistent with a NORMAL vol in price points
    no stable ratio         -> different convention entirely; read the table

Run on the Bloomberg Terminal machine, from the repo root:
    python diagnose_rate_vol.py
    python diagnose_rate_vol.py "FFZ6 Comdty"
"""
import datetime as dt
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))


def main(underlying="SFRZ6 Comdty"):
    from app.data.bloomberg import ASSET_DEFAULTS, get_provider, MockProvider, act365
    from app.data.chain_builder import (classify_asset, implied_vol_from_price,
                                        _f, _to_date)

    provider = get_provider(prefer_live=True)
    if isinstance(provider, MockProvider):
        print("MOCK provider -- no Terminal. Nothing to diagnose; run this on the "
              "Bloomberg machine.")
        return 1

    ac = classify_asset(underlying)
    print(f"underlying : {underlying}")
    print(f"asset class: {ac}")
    if ac != "RATES_PRICE":
        print("  (not a rate future -- diagnostic still runs, but the units "
              "issue this checks for is specific to rate futures)")

    spot = provider.spot(underlying)
    print(f"spot/px    : {spot}")

    members = provider.chain_tickers(underlying, call_put="C")
    print(f"chain      : {len(members)} members; sample {members[:3]}")
    if not members:
        print("No chain members. Stop here.")
        return 1

    df = provider.option_fields(members[:400])
    cols = {str(c).lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_iv = col("ivol_mid", "ivol")
    c_k = col("opt_strike_px", "strike")
    c_exp = col("opt_expire_dt", "expiry")
    c_pc = col("opt_put_call", "put_call")
    c_bid, c_ask, c_last = col("px_bid"), col("px_ask"), col("px_last")
    c_undl = col("opt_undl_px")

    print(f"columns    : {list(df.columns)[:10]}")
    if c_iv is None:
        print("\nNo IVOL_MID column at all -> the app already backs vol out of "
              "the mid price. Check the smile chart looks sane and stop here.")
        return 0

    as_of = dt.date.today()
    shift = float(ASSET_DEFAULTS[ac]["shift"])

    rows, ratios = [], []
    for tk, r in df.iterrows():
        K = _f(r.get(c_k)) if c_k else None
        exp = _to_date(r.get(c_exp)) if c_exp else None
        raw = _f(r.get(c_iv))
        if not K or not exp or exp <= as_of or raw is None or raw <= 0:
            continue
        T = act365(as_of, exp)
        F = _f(r.get(c_undl)) if c_undl else None
        F = F or spot

        bid = _f(r.get(c_bid)) if c_bid else None
        ask = _f(r.get(c_ask)) if c_ask else None
        last = _f(r.get(c_last)) if c_last else None
        mid = 0.5 * (bid + ask) if (bid and ask and bid > 0 and ask > 0) else (
            last if (last and last > 0) else None)
        if mid is None:
            continue

        pc_raw = str(r.get(c_pc)).strip().upper()[:1] if c_pc else "C"
        is_call = pc_raw in ("C", "1")

        backed = implied_vol_from_price(mid, F, K, T, is_call=is_call, shift=shift)
        if not backed or backed <= 0:
            continue

        rows.append((str(tk), K, "C" if is_call else "P", T, F, mid, raw, backed,
                     raw / backed))
        ratios.append(raw / backed)

    if not rows:
        print("\nCould not back any vols out of mid prices (no usable bid/ask). "
              "Check entitlements / that the market is open.")
        return 1

    # Show the strikes nearest the money.
    rows.sort(key=lambda x: abs(x[1] - (x[4] or spot)))
    print(f"\nBacked lognormal vol out of the mid price for {len(rows)} options.")
    print("Nearest-the-money sample:\n")
    print(f"  {'ticker':26s} {'K':>8s} {'cp':>3s} {'mid':>7s} "
          f"{'IVOL_MID':>9s} {'from price':>11s} {'ratio':>8s}")
    for tk, K, cp, T, F, mid, raw, backed, ratio in rows[:12]:
        print(f"  {tk[:26]:26s} {K:8.3f} {cp:>3s} {mid:7.4f} "
              f"{raw:9.4f} {backed:11.5f} {ratio:8.1f}")

    med = statistics.median(ratios)
    lo, hi = min(ratios), max(ratios)
    print(f"\nratio IVOL_MID / (vol implied by price): median {med:.1f} "
          f"(range {lo:.1f} to {hi:.1f})")

    # If IVOL_MID is quoted on the RATE rather than the price, the two are
    # linked through the common normal vol:
    #     sigma_rate * rate = sigma_normal = sigma_price * price
    # so the ratio lands near price/rate, NOT near 1 or 100.
    fwd_rate = 100.0 - spot
    rate_ratio = (spot / fwd_rate) if fwd_rate > 0 else float("inf")
    print(f"expected ratio if IVOL_MID is quoted on the RATE: "
          f"price/rate = {spot:.2f}/{fwd_rate:.2f} = {rate_ratio:.1f}")

    print("\n--- VERDICT ---")
    if 0.9 <= med <= 1.1:
        print("IVOL_MID is already a DECIMAL lognormal PRICE vol. Units fine.")
        print("If the smile still looks odd, the cause is elsewhere.")
    elif 90 <= med <= 110:
        print("IVOL_MID is a LOGNORMAL PERCENT on the PRICE (~x100).")
        print("The `if iv > 3.0` guess fails for rate futures (price vols < 3).")
        print("Fix: make the conversion asset-class-aware.")
    elif 0.6 * rate_ratio <= med <= 1.6 * rate_ratio:
        print(f"Ratio ~{med:.1f} is close to price/rate ({rate_ratio:.1f}).")
        print("=> IVOL_MID is a lognormal vol quoted on the RATE, not the price.")
        print("Units are being converted correctly; the vol simply refers to a")
        print("different underlying than the one we fit. Feeding it to a")
        print("price-space Black-76 SABR fit is meaningless.")
        print("Fix: for RATES_PRICE, ignore IVOL_MID and back the vol out of the")
        print("mid PRICE (convention-free).")
    elif 0.6 * spot <= med <= 1.6 * spot:
        print(f"Ratio ~{med:.0f} is close to F ({spot:.1f}): signature of a NORMAL")
        print("(Bachelier) vol in price points. Same fix: prefer the price back-out.")
    else:
        print(f"Ratio ~{med:.1f} matches no clean convention. Spread "
              f"({lo:.1f} to {hi:.1f}) matters: tight => units, wide => model.")
        print("Send the table above.")

    usable = len(rows)
    print(f"\nOptions with a usable mid price (the fix depends on this): {usable}")
    if usable < 8:
        print("  WARNING: too few to calibrate from prices alone. The back-out fix")
        print("  would leave the smile too sparse -- say so before we ship it.")
    else:
        print("  Enough to calibrate from prices alone.")

    print("\nThe price back-out column is convention-free and is the safest")
    print("source of truth for rate futures.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "SFRZ6 Comdty"))
