"""
Plain-English interpreter for the option-implied probability dashboard.

Same philosophy as the event parser: rule-based, no LLM, fully auditable.
Every sentence is assembled from the computed numbers, so the read can always
be traced back to the distribution and positioning payloads that produced it.

interpret(dist, positioning) -> {
    "headline":  one-sentence summary of the priced probability
    "sections":  [{"title", "text"}, ...] in reading order:
                 what is priced / positioning / the contrarian view / caveats
    "inputs":    the handful of numbers the text was built from (audit trail)
}

The interpreter never raises: missing fields degrade the text, not the call.
Output is a read of market pricing for internal research, not advice, and the
caveats section says so explicitly.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v, is_percent: bool, dp: int | None = None) -> str:
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if is_percent:
        return f"{v:.2f}%"
    a = abs(v)
    if dp is None:
        dp = 0 if a >= 1000 else (2 if a >= 10 else 4)
    return f"{v:,.{dp}f}"


def _pct(p) -> str:
    return f"{p * 100:.0f}%" if p is not None else "n/a"


def _odds_multiple(p: float) -> float | None:
    """Approximate payout multiple of a binary on the side with probability p."""
    if p is None or p <= 0.005 or p >= 0.995:
        return None
    return 1.0 / p


def _bucket(p: float) -> tuple[str, str]:
    """(bucket_id, phrase) describing how firmly the event is priced."""
    if p >= 0.95:
        return "near_certain", "priced as a near-certainty"
    if p >= 0.80:
        return "strong", "strongly priced in"
    if p >= 0.65:
        return "consensus", "a clear consensus lean"
    if p >= 0.55:
        return "mild", "a modest lean, little more than a tilted coin"
    if p >= 0.45:
        return "coin_toss", "essentially a coin toss"
    if p >= 0.35:
        return "mild_against", "a modest lean against"
    if p >= 0.20:
        return "out_of_favour", "out of favour but far from dismissed"
    if p >= 0.05:
        return "tail", "priced as a tail outcome"
    return "remote", "priced as remote"


def _event_phrase(dist: dict) -> str:
    u = dist.get("is_percent", False)
    d = dist.get("direction")
    thr = _fmt(dist.get("threshold"), u)
    if d == "above":
        return f"finishes above {thr}"
    if d == "below":
        return f"finishes below {thr}"
    if d == "between":
        return f"finishes between {thr} and {_fmt(dist.get('threshold_hi'), u)}"
    return f"meets the condition ({dist.get('condition', '?')})"


def _opposite_phrase(dist: dict) -> str:
    u = dist.get("is_percent", False)
    d = dist.get("direction")
    thr = _fmt(dist.get("threshold"), u)
    if d == "above":
        return f"finishing at or below {thr}"
    if d == "below":
        return f"finishing at or above {thr}"
    if d == "between":
        return (f"finishing outside the {thr} to "
                f"{_fmt(dist.get('threshold_hi'), u)} range")
    return "the opposite outcome"


def _structures(asset_class: str, rate_space: bool, contrarian_up: bool | None,
                between: bool) -> str:
    """Generic instrument language per asset class. Deliberately non-specific:
    this names the natural family of structures, not a trade ticket."""
    if between:
        return ("the natural expressions are the wings against the range: a "
                "strangle or the short body of an iron condor, depending on "
                "whether the range is being faded or owned")
    side = None
    if contrarian_up is not None:
        side = "upper" if contrarian_up else "lower"
    wing = f"the {side} wing" if side else "the out-of-favour wing"
    if asset_class in ("RATES", "RATES_PRICE"):
        return (f"out-of-the-money options on the same rate future expiry "
                f"({wing} in rate terms), or vertical spreads to cheapen the "
                f"premium at the cost of capping the payout")
    if asset_class == "FX":
        return (f"out-of-the-money options or a risk reversal favouring "
                f"{wing}, which also monetises the skew if that wing is bid")
    if asset_class == "CMDTY":
        return f"out-of-the-money options or call/put spreads on {wing}"
    return (f"out-of-the-money options or option spreads on {wing} of the "
            f"index smile")


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _sec_priced(dist: dict) -> str:
    p = dist.get("probability")
    u = dist.get("is_percent", False)
    fwd = dist.get("forward")
    stats = dist.get("stats") or {}
    std = stats.get("std")
    thr = dist.get("threshold")
    _, phrase = _bucket(p)

    bits = [f"At {_pct(p)}, the outcome is {phrase}."]

    if fwd is not None and thr is not None and std:
        z = (thr - fwd) / std
        rel = "above" if z > 0 else "below"
        if abs(z) < 0.25:
            bits.append(
                f"The threshold sits almost on top of the forward "
                f"({_fmt(fwd, u)}), only {abs(z):.1f} standard deviations "
                f"{rel} it, so the probability is acutely sensitive to small "
                f"moves in the underlying and will swing with the tape.")
        else:
            bits.append(
                f"The threshold is {abs(z):.1f} standard deviations {rel} the "
                f"forward of {_fmt(fwd, u)} (distribution std "
                f"{_fmt(std, u)}), so this is a question about the "
                f"{'right' if z > 0 else 'left'} part of the distribution "
                f"rather than the central case.")

    mean, p05, p95 = stats.get("mean"), stats.get("p05"), stats.get("p95")
    if mean is not None and p05 is not None and p95 is not None:
        bits.append(
            f"The market's central expectation is {_fmt(mean, u)}, with 90% "
            f"of the risk-neutral weight between {_fmt(p05, u)} and "
            f"{_fmt(p95, u)}.")

    sabr = dist.get("sabr") or {}
    rho = sabr.get("rho")
    if rho is not None and abs(rho) >= 0.4:
        wing = "lower" if rho < 0 else "upper"
        bits.append(
            f"The smile is steeply skewed (rho {rho:.2f}): the {wing} wing "
            f"is bid, meaning the market pays up for protection in that "
            f"direction.")
    return " ".join(bits)


def _sec_positioning(dist: dict, pos: dict | None) -> str:
    p = dist.get("probability")
    u = dist.get("is_percent", False)
    event = _event_phrase(dist)

    lean = (f"With {_pct(p)} of the risk-neutral weight on the underlying "
            f"{event.replace('finishes', 'finishing')}, "
            if p is not None else "")
    if p is not None and p >= 0.55:
        lean += "the options market appears largely positioned for that outcome."
    elif p is not None and p <= 0.45:
        lean += ("most participants appear positioned for the opposite, and "
                 "the event itself is the minority view.")
    else:
        lean += "positioning via pricing alone is close to balanced."

    if not pos or not isinstance(pos, dict):
        return (lean + " No strike-level positioning history is available for "
                "this expiry yet (accumulate snapshots or run /api/backfill), "
                "so this read rests on pricing alone.")

    bits = [lean]
    sm = pos.get("summary") or {}
    rate_space = bool(pos.get("rate_space"))
    fwd = pos.get("forward")

    pc = sm.get("put_call_oi_ratio")
    if pc is not None:
        if pc >= 1.3:
            tilt = "put-heavy, consistent with hedged or outright bearish books"
        elif pc <= 0.7:
            tilt = "call-heavy, consistent with upside-seeking books"
        else:
            tilt = "roughly balanced between puts and calls"
        note = (" (contract types are quoted in price space here: a call on "
                "the future is a position for lower rates)" if rate_space else "")
        bits.append(f"Open interest at this expiry is {tilt} "
                    f"(put/call OI ratio {pc:.2f}){note}.")

    cog = sm.get("oi_center_of_gravity")
    if cog is not None and fwd is not None:
        rel = "below" if cog < fwd else "above"
        bits.append(f"The open-interest centre of gravity sits at "
                    f"{_fmt(cog, u)}, {rel} the forward of {_fmt(fwd, u)}"
                    f"{', i.e. the crowd is camped ' + rel + ' current pricing' if abs(cog - fwd) > 1e-9 else ''}.")

    mp = sm.get("max_pain")
    if mp is not None:
        bits.append(f"Max pain is {_fmt(mp, u)}.")

    top = sm.get("top_conviction") or []
    strong = [t for t in top if isinstance(t, dict)
              and t.get("composite") in ("high", "moderate")]
    if strong:
        descr = []
        for t in strong[:3]:
            d = t.get("direction") or 0
            if rate_space:
                way = "lower-rate" if d > 0 else ("higher-rate" if d < 0 else "flat")
            else:
                way = "bullish" if d > 0 else ("bearish" if d < 0 else "flat")
            descr.append(f"{_fmt(t.get('strike'), u)} ({way} build, "
                         f"{t.get('n_agree', 0)}/3 signals agreeing)")
        bits.append("The highest-conviction strikes are " + "; ".join(descr) + ".")
    if not pos.get("deltas_available"):
        bits.append("Conviction deltas are still building history, so the "
                    "strike-level read is tentative.")
    return " ".join(bits)


def _sec_contrarian(dist: dict) -> str:
    p = dist.get("probability")
    if p is None:
        return "No probability computed, so no contrarian view can be framed."
    d = dist.get("direction")
    between = d == "between"
    event = _event_phrase(dist)
    opp = _opposite_phrase(dist)
    ac = dist.get("asset_class", "")
    rate_space = bool(dist.get("rate_space"))

    # Which side is contrarian, and which wing does it sit on?
    if p >= 0.5:
        c_p = 1.0 - p                     # contrarian = event fails
        c_side = opp
        contrarian_up = (d == "below")    # fade a "below" call = own the upside
    else:
        c_p = p                           # contrarian = event happens
        c_side = event.replace("finishes", "finishing")
        contrarian_up = (d == "above")
    if between:
        contrarian_up = None

    mult = _odds_multiple(c_p)
    payoff = (f"a binary on that side is priced near {_pct(c_p)}, so it pays "
              f"roughly {mult:.1f} to 1 if it lands"
              if mult is not None else
              "that side is priced so far out that listed structures offer "
              "extreme but illiquid odds")

    structures = _structures(ac, rate_space, contrarian_up, between)

    if 0.45 <= p <= 0.55:
        return (f"With pricing near 50/50 there is no crowded side to fade: "
                f"the contrarian angle here is not direction but price of "
                f"volatility, i.e. owning or selling the move itself. If a "
                f"directional view is held anyway, either side pays roughly "
                f"even money via {structures}.")

    return (f"The consensus side ({_pct(p)}) offers little payout for being "
            f"right. The contrarian position is the underlying {c_side}: "
            f"{payoff}. Natural expressions would be {structures}. The case "
            f"for it, if one holds it, is precisely that the market has "
            f"largely stopped paying for that outcome.")


def _sec_caveats(dist: dict) -> str:
    bits = ["These are risk-neutral probabilities: they embed risk premia and "
            "are not physical forecasts, and the gap matters most for rates "
            "and index tails."]
    t = dist.get("target_date")
    e = dist.get("expiry")
    if t and e and str(t)[:10] != str(e)[:10]:
        bits.append(f"The condition's target date ({t}) was mapped to the "
                    f"nearest listed expiry ({e}); the probability answers "
                    f"'at that expiry', not 'at any point before'.")
    else:
        bits.append("The probability answers 'at expiry', not 'touches at "
                    "any point before'.")
    sabr = dist.get("sabr") or {}
    rmse = sabr.get("rmse")
    if rmse is not None and rmse > 0.02:
        bits.append(f"The SABR fit RMSE is {rmse * 100:.1f} vol points, on "
                    f"the high side, so treat the tails with extra care.")
    bits.append("This is an automated, rule-based read of market pricing for "
                "internal research, not investment advice.")
    return " ".join(bits)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def interpret(dist: dict, positioning: dict | None = None) -> dict:
    """Assemble the plain-English assessment. Never raises."""
    try:
        p = dist.get("probability")
        headline = (f"Options imply a {_pct(p)} probability that "
                    f"{dist.get('underlying', 'the underlying')} "
                    f"{_event_phrase(dist)} at the {dist.get('expiry', '?')} "
                    f"expiry.")
        sections = [
            {"title": "What is priced", "text": _sec_priced(dist)},
            {"title": "Positioning", "text": _sec_positioning(dist, positioning)},
            {"title": "The contrarian view", "text": _sec_contrarian(dist)},
            {"title": "Caveats", "text": _sec_caveats(dist)},
        ]
        stats = dist.get("stats") or {}
        return {
            "headline": headline,
            "sections": sections,
            "inputs": {
                "probability": p,
                "forward": dist.get("forward"),
                "threshold": dist.get("threshold"),
                "std": stats.get("std"),
                "rho": (dist.get("sabr") or {}).get("rho"),
                "put_call_oi_ratio": ((positioning or {}).get("summary") or {}).get("put_call_oi_ratio"),
                "positioning_used": bool(positioning),
            },
        }
    except Exception as exc:  # degrade, never break the main response
        return {"headline": "Interpretation unavailable.",
                "sections": [{"title": "Error",
                              "text": f"Interpreter failed: {exc}"}],
                "inputs": {}}
