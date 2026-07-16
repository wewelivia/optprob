"""
Local strike-level positioning snapshot store (SQLite).

WHY THIS EXISTS
---------------
Bloomberg's *history* for granular option-chain data (open interest and volume
by individual strike/expiry, not just index-level aggregates) is patchy --
volume history at the strike level rarely goes back far, and strikes roll on and
off as the underlying moves. So for the conviction panel we cannot rely on a
single ``bdh`` call to tell us "what was OI at this exact strike 5 days ago".

Instead we own the strike-level history: every time the tool builds a live
chain we write the current snapshot (OI, volume, IV, mid) here with a timestamp,
keyed by (as_of, underlying, expiry, strike, call_put). The 1-day and 5-day
deltas are then computed from *our own* complete, consistent series. Longer
20/60-day windows and index-level series come from Bloomberg ``bdh`` (see
``bloomberg.py``), and a one-time backfill can seed this store so we don't wait
60 days for a usable long window.

This mirrors the SQLite pattern already used by the call-performance tracker.

The store is intentionally dependency-light (stdlib ``sqlite3`` only) and safe
to import even when the rest of the Bloomberg stack is unavailable.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional


# Default location: alongside the backend, in a data/ dir. Override with env.
_DEFAULT_DB = os.environ.get(
    "POSITIONING_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 "data", "positioning.db"),
)


@dataclass
class SnapshotRow:
    as_of: dt.date
    underlying: str
    expiry: dt.date
    strike: float
    call_put: str            # 'C' | 'P'
    open_interest: Optional[float]
    volume: Optional[float]
    implied_vol: Optional[float]
    mid_price: Optional[float]

    def key(self) -> tuple:
        return (self.as_of.isoformat(), self.underlying,
                self.expiry.isoformat(), float(self.strike), self.call_put)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS option_snapshot (
    as_of         TEXT    NOT NULL,   -- ISO date of the snapshot
    underlying    TEXT    NOT NULL,
    expiry        TEXT    NOT NULL,   -- ISO date
    strike        REAL    NOT NULL,
    call_put      TEXT    NOT NULL,   -- 'C' | 'P'
    open_interest REAL,
    volume        REAL,
    implied_vol   REAL,
    mid_price     REAL,
    inserted_at   TEXT    NOT NULL,
    PRIMARY KEY (as_of, underlying, expiry, strike, call_put)
);
CREATE INDEX IF NOT EXISTS ix_snap_lookup
    ON option_snapshot (underlying, expiry, strike, call_put, as_of);
CREATE INDEX IF NOT EXISTS ix_snap_asof
    ON option_snapshot (underlying, as_of);
"""


class PositioningStore:
    """Thin SQLite wrapper for writing and reading strike-level snapshots."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or _DEFAULT_DB
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writes -------------------------------------------------------------
    def write_snapshot(self, rows: Iterable[SnapshotRow],
                       replace: bool = True) -> int:
        """Persist a batch of snapshot rows. Same-day re-runs upsert (replace)
        by default so the latest intraday pull wins for that date."""
        now = dt.datetime.utcnow().isoformat(timespec="seconds")
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        sql = (f"{verb} INTO option_snapshot "
               "(as_of, underlying, expiry, strike, call_put, open_interest, "
               " volume, implied_vol, mid_price, inserted_at) "
               "VALUES (?,?,?,?,?,?,?,?,?,?)")
        payload = [
            (r.as_of.isoformat(), r.underlying, r.expiry.isoformat(),
             float(r.strike), r.call_put,
             _fnone(r.open_interest), _fnone(r.volume),
             _fnone(r.implied_vol), _fnone(r.mid_price), now)
            for r in rows
        ]
        if not payload:
            return 0
        with self._conn() as c:
            c.executemany(sql, payload)
        return len(payload)

    # -- reads --------------------------------------------------------------
    def available_dates(self, underlying: str,
                        before: dt.date | None = None) -> list[dt.date]:
        """Distinct snapshot dates we hold for an underlying (ascending)."""
        q = "SELECT DISTINCT as_of FROM option_snapshot WHERE underlying=?"
        args: list = [underlying]
        if before is not None:
            q += " AND as_of<=?"
            args.append(before.isoformat())
        q += " ORDER BY as_of ASC"
        with self._conn() as c:
            return [dt.date.fromisoformat(r[0]) for r in c.execute(q, args)]

    def snapshot_on(self, underlying: str, as_of: dt.date,
                    expiry: dt.date | None = None) -> list[SnapshotRow]:
        """All rows on a given date (optionally one expiry)."""
        q = ("SELECT * FROM option_snapshot WHERE underlying=? AND as_of=?")
        args: list = [underlying, as_of.isoformat()]
        if expiry is not None:
            q += " AND expiry=?"
            args.append(expiry.isoformat())
        with self._conn() as c:
            return [_row_to_snap(r) for r in c.execute(q, args)]

    def series_for_strike(self, underlying: str, expiry: dt.date,
                          strike: float, call_put: str,
                          lookback_days: int = 90) -> list[SnapshotRow]:
        """Time series for one (expiry,strike,cp) over the recent window."""
        cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
        q = ("SELECT * FROM option_snapshot WHERE underlying=? AND expiry=? "
             "AND strike=? AND call_put=? AND as_of>=? ORDER BY as_of ASC")
        with self._conn() as c:
            rows = c.execute(q, [underlying, expiry.isoformat(),
                                 float(strike), call_put, cutoff])
            return [_row_to_snap(r) for r in rows]

    def nearest_prior_date(self, underlying: str, target_gap: int,
                           ref: dt.date | None = None) -> dt.date | None:
        """Find the stored date closest to `target_gap` calendar days before
        `ref` (default today). Used to pin the '5-day-ago' snapshot to a real
        trading day we actually hold, tolerating weekends/holidays."""
        ref = ref or dt.date.today()
        want = ref - dt.timedelta(days=target_gap)
        dates = self.available_dates(underlying, before=ref)
        if not dates:
            return None
        return min(dates, key=lambda d: abs((d - want).days))


def _fnone(x) -> float | None:
    try:
        if x is None:
            return None
        xf = float(x)
        # store NaN as NULL
        return None if xf != xf else xf
    except (TypeError, ValueError):
        return None


def _row_to_snap(r: sqlite3.Row) -> SnapshotRow:
    return SnapshotRow(
        as_of=dt.date.fromisoformat(r["as_of"]),
        underlying=r["underlying"],
        expiry=dt.date.fromisoformat(r["expiry"]),
        strike=float(r["strike"]),
        call_put=r["call_put"],
        open_interest=r["open_interest"],
        volume=r["volume"],
        implied_vol=r["implied_vol"],
        mid_price=r["mid_price"],
    )


def rows_from_chain(chain) -> list[SnapshotRow]:
    """Flatten an OptionChain into snapshot rows for persistence."""
    out: list[SnapshotRow] = []
    for e in chain.expiries:
        for q in e.quotes:
            out.append(SnapshotRow(
                as_of=chain.as_of,
                underlying=chain.underlying,
                expiry=e.expiry,
                strike=float(q.strike),
                call_put=q.call_put,
                open_interest=q.open_interest,
                volume=q.volume,
                implied_vol=q.implied_vol,
                mid_price=q.mid_price,
            ))
    return out
