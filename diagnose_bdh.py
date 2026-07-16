"""
Dump the ACTUAL shape xbbg's bdh() returns on this machine.

Why: the backfill 500 ("cannot insert date, already exists") was caused by
bdh_fields assuming xbbg 0.x's DatetimeIndex + MultiIndex-column shape. The
Rust/Arrow-backed v1 line returns something different. normalise_bdh() now
handles four plausible shapes, but this script confirms which one you actually
get, so we're reading the environment rather than guessing at it.

Run on the Bloomberg Terminal machine, from the repo root:
    python diagnose_bdh.py
"""
import datetime as dt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))


def main():
    try:
        from xbbg import blp
    except Exception as e:
        print(f"xbbg import failed: {e}")
        return 1

    try:
        import xbbg
        print(f"xbbg version: {getattr(xbbg, '__version__', 'unknown')}")
    except Exception:
        pass

    end = dt.date.today()
    start = end - dt.timedelta(days=10)

    # A liquid, definitely-entitled pair so a blank result means shape/plumbing,
    # not entitlements.
    tickers = ["SPX Index", "SPY US Equity"]
    flds = ["PX_LAST", "PX_VOLUME"]

    print(f"\nbdh(tickers={tickers}, flds={flds}, {start} -> {end})")
    raw = blp.bdh(tickers=tickers, flds=flds, start_date=start, end_date=end)

    print("\n--- RAW ---")
    print(f"type            : {type(raw)}")
    print(f"module          : {type(raw).__module__}")
    print(f"shape           : {getattr(raw, 'shape', 'n/a')}")
    print(f"has .to_native  : {hasattr(raw, 'to_native')}")
    print(f"has .to_pandas  : {hasattr(raw, 'to_pandas')}")

    from app.data.bloomberg import _to_pandas, normalise_bdh
    df = _to_pandas(raw)
    print("\n--- AFTER _to_pandas ---")
    print(f"type            : {type(df)}")
    print(f"index type      : {type(getattr(df, 'index', None))}")
    print(f"index dtype     : {getattr(getattr(df, 'index', None), 'dtype', 'n/a')}")
    print(f"index name      : {getattr(getattr(df, 'index', None), 'name', 'n/a')}")
    print(f"columns type    : {type(getattr(df, 'columns', None))}")
    print(f"columns         : {list(getattr(df, 'columns', []))[:8]}")
    print(f"'date' a column?: {'date' in [str(c).lower() for c in getattr(df, 'columns', [])]}")
    print("\nhead:")
    print(df.head(3))

    print("\n--- AFTER normalise_bdh ---")
    long = normalise_bdh(df, tickers, flds)
    print(f"rows            : {len(long)}")
    print(f"columns         : {list(long.columns)}")
    print(f"tickers seen    : {sorted(set(long['ticker']))}")
    print(f"fields seen     : {sorted(set(long['field']))}")
    print(f"dates seen      : {len(set(long['date']))}")
    print("\nhead:")
    print(long.head(6))

    if long.empty:
        print("\nEMPTY -> shape not handled. Send the RAW block above.")
        return 1
    print("\nOK: bdh normalises cleanly. Backfill should work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
