"""
Offline tests for the plain-English interpreter. No Bloomberg, no network.

Run from the project root:  python tests/test_interpreter.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.interpreter import interpret  # noqa: E402


def _dist(p=0.68, direction="below", threshold=4.0, forward=4.12, std=0.45,
          is_percent=True, rate_space=True, asset_class="RATES_PRICE",
          rho=-0.2, rmse=0.004):
    return {
        "underlying": "FEDFUNDS", "asset_class": asset_class,
        "source": "bloomberg", "expiry": "2026-12-18", "T": 0.42,
        "forward": forward, "rate_space": rate_space, "is_percent": is_percent,
        "sabr": {"alpha": 0.3, "beta": 0.5, "rho": rho, "nu": 0.8,
                 "shift": 0.0, "rmse": rmse},
        "stats": {"forward": forward, "mean": forward - 0.02, "mode": forward,
                  "median": forward, "std": std,
                  "p05": forward - 1.6 * std, "p25": forward - 0.6 * std,
                  "p75": forward + 0.6 * std, "p95": forward + 1.6 * std},
        "probability": p, "condition": f"below {threshold}% by December",
        "direction": direction, "threshold": threshold, "threshold_hi": None,
        "target_date": "2026-12-31", "complement": 1 - p, "odds": "x",
    }


def _positioning(pc_ratio=1.45, cog=3.85, max_pain=4.00, deltas=True):
    return {
        "rate_space": True, "forward": 4.12, "deltas_available": deltas,
        "history_days": 30 if deltas else 1,
        "summary": {
            "put_call_oi_ratio": pc_ratio,
            "oi_center_of_gravity": cog,
            "max_pain": max_pain,
            "top_conviction": [
                {"strike": 3.75, "call_put": "C", "composite": "high",
                 "direction": 1, "magnitude": 2.1, "n_agree": 3},
                {"strike": 4.25, "call_put": "P", "composite": "moderate",
                 "direction": -1, "magnitude": 1.4, "n_agree": 2},
            ],
        },
    }


def test_structure():
    out = interpret(_dist(), _positioning())
    assert set(out) == {"headline", "sections", "inputs"}
    titles = [s["title"] for s in out["sections"]]
    assert titles == ["What is priced", "Positioning",
                      "The contrarian view", "Caveats"]
    assert "68%" in out["headline"] and "4.00%" in out["headline"]
    assert out["inputs"]["positioning_used"] is True
    print("ok  structure + headline")


def test_consensus_lean_reads_contrarian_other_side():
    out = interpret(_dist(p=0.82), _positioning())
    text = " ".join(s["text"] for s in out["sections"])
    assert "strongly priced" in text
    assert "at or above 4.00%" in text, "contrarian side must be the fade"
    assert "to 1" in text, "payout odds must be quantified"
    print("ok  high-probability contrarian read")


def test_tail_event_contrarian_is_the_event():
    out = interpret(_dist(p=0.08), None)
    text = " ".join(s["text"] for s in out["sections"])
    assert "tail" in text
    assert "finishing below 4.00%" in text
    assert "pricing alone" in text, "must flag missing positioning"
    print("ok  tail event + no positioning fallback")


def test_coin_toss_has_no_crowded_side():
    out = interpret(_dist(p=0.5), None)
    text = next(s["text"] for s in out["sections"]
                if s["title"] == "The contrarian view")
    assert "no crowded side" in text
    print("ok  coin toss")


def test_threshold_near_forward_flagged():
    out = interpret(_dist(forward=4.02, std=0.45), None)
    text = next(s["text"] for s in out["sections"]
                if s["title"] == "What is priced")
    assert "acutely sensitive" in text
    print("ok  near-forward sensitivity flag")


def test_rate_space_direction_language():
    out = interpret(_dist(), _positioning())
    text = next(s["text"] for s in out["sections"]
                if s["title"] == "Positioning")
    assert "lower-rate" in text and "higher-rate" in text, \
        "rate-space conviction must speak in rate terms, not call/put"
    assert "put/call OI ratio 1.45" in text
    print("ok  rate-space positioning language")


def test_between_condition():
    d = _dist(direction="between", threshold=200.0, is_percent=False,
              rate_space=False, asset_class="EQUITY", forward=220.0, std=25.0)
    d["threshold_hi"] = 240.0
    out = interpret(d, None)
    assert "between 200" in out["headline"]
    text = next(s["text"] for s in out["sections"]
                if s["title"] == "The contrarian view")
    assert "strangle" in text or "condor" in text
    print("ok  between condition")


def test_never_raises_on_garbage():
    out = interpret({}, {"summary": "not-a-dict-value"})
    assert "headline" in out and "sections" in out
    print("ok  degrades on malformed input")


if __name__ == "__main__":
    test_structure()
    test_consensus_lean_reads_contrarian_other_side()
    test_tail_event_contrarian_is_the_event()
    test_coin_toss_has_no_crowded_side()
    test_threshold_near_forward_flagged()
    test_rate_space_direction_language()
    test_between_condition()
    test_never_raises_on_garbage()
    print("\nall interpreter tests passed\n")

    # Show the full worked example for the strategist's eye
    out = interpret(_dist(), _positioning())
    print(out["headline"] + "\n")
    for s in out["sections"]:
        print(f"[{s['title']}]")
        print(s["text"] + "\n")
