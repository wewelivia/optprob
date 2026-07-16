# Project Handover — Option-Implied Probability Dashboard

> **Purpose of this doc.** Everything an LLM (or engineer) needs to continue this
> project cold, with no prior conversation context. Read this top-to-bottom before
> touching code. It states what the tool does, how it is wired, the exact file map,
> the conviction/positioning design decisions and *why* they were made, how to run
> and test it, and the known caveats.

---

## 1. What this is

A dashboard that extracts the **risk-neutral (option-implied) probability distribution**
of an underlying from the listed options market and answers event-probability questions,
e.g. *"What is the probability SPX is above 6000 by December?"* or *"...Fed funds above
5% by December?"*.

**Method pipeline:**

```
Bloomberg (xbbg / blpapi)  →  Option chain  →  SABR smile fit (per expiry)
   → Breeden–Litzenberger on the FITTED call-price curve → risk-neutral density
   → integrate → CDF → P(event)
```

Without a Bloomberg Terminal the app serves **realistic synthetic surfaces** (badged in
the UI) so it is fully explorable anywhere. On the Terminal machine it uses live data
automatically — no code changes.

On top of the probability engine sits a **conviction / positioning layer** (the most
recent work) that reads *where in the option chain participants are actually positioned*
and how strongly, so a probability read can be judged against real money.

---

## 2. Repository map

Project root: `rnd_dashboard/`
(User's local copy: `/Users/julienlafargue/Documents/HouseView/…`; sandbox copy:
`/home/user/workspace/rnd_dashboard/`.)

```
rnd_dashboard/
├── README.md                       # user-facing overview + quick start
├── HANDOVER.md                     # THIS FILE
├── requirements.txt                # pip deps (blpapi/xbbg commented; Terminal-only)
├── dashboard_preview.png
├── backend/
│   └── app/
│       ├── main.py                 # FastAPI app + all routes + static mount
│       ├── core/
│       │   ├── event_parser.py     # parse "above 6000 by December" -> threshold, side, target date
│       │   ├── sabr.py             # SABR (Hagan/Obløj), shifted SABR, per-expiry WLS fit
│       │   ├── breeden_litzenberger.py  # Black-76 pricing + q(K)=e^{rT}·∂²C/∂K²
│       │   ├── service.py          # orchestration: chain -> dist -> P(event); positioning; backfill
│       │   └── conviction.py       # ***the conviction/positioning engine*** (see §5)
│       ├── data/
│       │   ├── bloomberg.py        # BloombergProvider (xbbg/blpapi) + MockProvider; bdh_fields()
│       │   ├── chain_builder.py    # provider-agnostic OptionChain assembly; IV-from-price; backfill rows
│       │   └── positioning_store.py# ***SQLite snapshot store*** for strike-level OI/vol/IV/mid (see §4)
│       ├── models/schemas.py       # pydantic request/response models
│       └── api/                    # (package placeholder)
├── frontend/
│   ├── index.html                  # single-page UI; includes the "Where the conviction is" section
│   ├── app.js                      # fetch + Plotly render; renderPositioning() (see §6)
│   └── style.css                   # incl. .pos-* and .chip styles
├── notebooks/
│   └── sabr_validation.ipynb
└── tests/
    ├── test_core_math.py           # SABR + Breeden–Litzenberger numerics
    ├── test_pipeline.py            # end-to-end chain->distribution
    ├── test_xbbg_shapes.py         # xbbg/narwhals dataframe-shape handling
    └── test_conviction.py          # ***conviction engine + store tests*** (see §7)
```

---

## 3. Run & test

```bash
cd rnd_dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd backend
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```

**IMPORTANT (Terminal machine):** the project folder is **OneDrive-synced**, so
uvicorn `--reload` sometimes misses saves. After editing, do a **hard restart**
(Ctrl+C then relaunch), not just a reload.

Tests (run from repo root):

```bash
python tests/test_core_math.py
python tests/test_pipeline.py
python tests/test_xbbg_shapes.py
python tests/test_conviction.py
```

Each test file is a plain script (prints "ALL … PASSED") and adds `backend/` to
`sys.path` at the top. All four currently pass.

---

## 4. Local snapshot store — `data/positioning_store.py`

**Why it exists.** Bloomberg's *strike-level* historical OI/volume is patchy. So the
tool persists its own daily snapshot of the chain locally and computes deltas from its
own accumulated history. (Same pattern as the call-performance-tracker's SQLite store.)

- **`PositioningStore`** — SQLite. DB path from env `POSITIONING_DB`, else
  `backend/data/positioning.db`.
- Table `option_snapshot`, primary key `(as_of, underlying, expiry, strike, call_put)`,
  columns `open_interest, volume, implied_vol, mid_price`.
- **`SnapshotRow`** dataclass mirrors those columns.
- Key methods:
  - `write_snapshot(rows)` — upsert (idempotent per PK).
  - `available_dates(underlying)` / `snapshot_on(underlying, date, expiry=…)`.
  - `series_for_strike(underlying, expiry, strike, call_put, lookback_days)`.
  - `nearest_prior_date(underlying, k, ref)` — pin the "k-days-ago" comparison to the
    nearest actually-stored trading day (handles weekends/holidays/gaps).
- Helper **`rows_from_chain(chain)`** converts a built `OptionChain` into `SnapshotRow`s.

**How it gets populated:**
1. **Every live run** — `service._get_chain()` persists the current chain snapshot
   (wrapped in try/except so a store failure never breaks the probability path).
2. **One-time backfill** — `POST /api/backfill` pulls `bdh` history (OPEN_INT, IVOL_MID,
   PX_LAST, PX_VOLUME) and seeds the store so deltas work immediately instead of waiting
   days for live accumulation.

---

## 5. Conviction engine — `core/conviction.py`  (the core design)

**Goal (user's spec, authoritative):** judge *conviction* per strike from the confluence
of three corroborating signals, each **z-scored against its OWN recent volatility** — so
a raw move means the same thing in a quiet summer as around an FOMC meeting.

### Three signals per strike
1. **ΔOI** — open-interest build (positioning base).
2. **volume / OI** — turnover *intensity* (fresh flow vs stale open interest).
3. **ΔIV** — pricing confirmation (the market's own price on the view).

### `ConvictionConfig` defaults
| field | default | meaning |
|---|---|---|
| `short_window` | 5 | operational signal you act on |
| `trend_window` | 20 | trend confirmation |
| `context_window` | 60 | z-score normalization horizon |
| `w_iv` | 0.45 | IV change weighted **highest** (market's own price on the view) |
| `w_voloi` | 0.35 | turnover second (fresh vs stale) |
| `w_oi` | 0.20 | raw OI change **lowest** (most prone to mechanical/rebalancing flow) |
| `high_conviction_magnitude` | 1.0 | |magnitude| gate for a High flag |
| `min_agreement` | 2 | ≥2 of 3 signals must agree in direction |
| `conflict_magnitude` | 1.0 | large but disagreeing ⇒ "conflicting" |
| `min_context_points` | 8 | need ≥8 observations to z-score at all |

### Scoring logic (three-stage, deliberately NOT a single blended number)
1. **Magnitude** = weighted sum of the three z-scores (weights above).
2. **Agreement flag** = count of signals pointing the same direction — displayed
   *alongside*, **not** folded into magnitude.
3. **Composite (gated)**:
   - **High** — magnitude ≥ gate **AND** ≥2 of 3 agree.
   - **Moderate** — magnitude ≥ gate/2 AND agreement OK.
   - **Conflicting** — strong signals that *oppose* each other (≥2 signals with
     |z| ≥ `conflict_magnitude` pointing opposite ways). This is surfaced as its own
     flag and **never averaged into a misleading "moderate."**
   - **Low** — everything else.
   - **n/a** — insufficient history to z-score (`< min_context_points`).

### Key internal helpers (careful — these encode bug fixes)
- **`_robust_scale(series)`** — robust volatility estimate **with a floor** so a smooth
  monotone series doesn't yield ~0 scale and explode the z-score.
- **`_zscore_change(change, hist, min_points, horizon)`** — scores a *k-day* change
  against the volatility of *k-day changes* (horizon-aware), not against level vol.
- **`_intensity_score(value, hist, min_points)`** — vol/OI scored as **elevation above a
  quiet baseline percentile**, NOT self-demeaned (persistent high turnover must read
  positive, not average away to ~0).
- **`_composite(...)`** — conflict detected from raw *opposed strong signals*.

### Public API
- **`ConvictionEngine(store, config).compute_for_expiry(underlying, expiry, as_of=…)`**
  → list of `StrikeConviction` (fields incl. `strike, call_put, oi, magnitude,
  direction, n_agree, z_oi, z_voloi, z_iv, composite`).
- **`summarize_positioning(rows, expiry, results, multiplier)`** → `PositioningSummary`
  with `put_call_oi_ratio`, `oi_center_of_gravity`, `max_pain`, `total_premium_notional`
  (= Σ OI·mid·multiplier), `total_call_oi`, `total_put_oi`, `top_conviction` (list).
- **`_max_pain(...)`** — standard max-pain strike.

Multiplier by asset class lives in `service._MULTIPLIER`
(`EQ_INDEX/EQUITY=100, FX=1, …`).

---

## 6. API surface (`main.py`)

All JSON responses pass through a NaN-safe `_scrub` so they never 500 on `nan/inf`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/presets` | example conditions |
| GET | `/api/diagnose?underlying=` | provider/chain diagnostics |
| GET | `/api/chain?underlying=&prefer_live=` | built option chain |
| POST | `/api/distribution` | body = `ProbabilityRequest` → density/CDF/P(event) |
| GET | `/api/positioning?underlying=&condition=&expiry=&short_window=&trend_window=&context_window=&prefer_live=` | per-strike conviction + summary for the expiry matching the condition's target date |
| POST | `/api/backfill?underlying=&days=90&prefer_live=` | one-time store seed from bdh history |
| GET | `/` | static frontend |

**`/api/positioning` response keys:** `expiry`, `history_days`, `deltas_available`,
`strikes` (list of per-strike dicts: `strike, call_put, oi, magnitude, direction,
n_agree, z_oi, z_voloi, z_iv, composite`), `summary` (the `PositioningSummary` dict incl.
`top_conviction`).

Example:
```
GET  /api/positioning?underlying=SPX%20Index&condition=above%208000%20by%20December
POST /api/backfill?underlying=SPX%20Index&days=90
```

---

## 7. Frontend (`frontend/app.js`, `index.html`, `style.css`)

- `run()` calls `/api/distribution`, renders PDF/CDF/smile/stats, then calls
  **`loadPositioning(underlying, condition, dist)`**, which hits `/api/positioning` and
  renders the **"Where the conviction is"** section. Positioning is *supplementary* — if
  it fails it is hidden and never blocks the main probability result.
- **`renderPositioning(p, dist)`** draws:
  1. **OI-by-strike bar chart** colored by conviction (`CONV_COLOR`), with the RND PDF
     overlaid on a secondary y-axis and the threshold line.
  2. **Signed-magnitude bar chart** (direction × magnitude; ↑ bullish build, ↓ bearish).
  3. **Positioning summary table** (P/C ratio, CoG, max pain, premium notional, call/put OI).
  4. **Top conviction table** (strike, colored read chip, magnitude, agreement n/3, direction arrow).
- Colors: high `#38d39f`, moderate `#4da3ff`, conflicting `#ff5c7a`, low `#5b6577`, na `#3a4255`.
- **Field-name gotchas the JS relies on:** per-strike open interest key is **`oi`**
  (not `open_interest`); agreement count is **`n_agree`**; summary exposes
  **`total_call_oi` / `total_put_oi`** (there is no single `total_oi`).

---

## 8. Tests — `tests/test_conviction.py`

Covers: `_robust_scale` no-explosion on smooth series; `_intensity_score` not
de-meaned; scoring (confluence→high, opposed→conflicting, noise→low); summary stats;
store round-trip + `nearest_prior_date` pinning. Seeds a synthetic multi-day store
(building call OI + rising turnover + rising IV ⇒ High; OI-up-but-IV-down ⇒ Conflicting;
flat noise ⇒ Low). Keep the other three suites green when changing shared code.

---

## 9. Design decisions & rationale (don't silently undo these)

- **Z-score against each signal's own recent volatility** matters more than the window
  length itself — the whole point is comparability across calm vs event regimes.
- **Weights** IV > vol/OI > raw OI, because raw OI is the most contaminated by
  mechanical/rebalancing flow and IV is the market pricing the view directly.
- **Agreement is displayed, not blended.** A big magnitude with internal disagreement is
  a *conflicting* signal, not a moderate one — averaging it away would mislead.
- **Local SQLite for strike-level history** because Bloomberg granular OI history is
  patchy; bdh backfill only seeds it.
- **Snapshot persistence is best-effort** (try/except) so it can never break the
  probability path.

---

## 10. Known caveats / gotchas

- **Conviction needs history.** With `< min_context_points` (8) snapshots, strikes read
  **n/a / low** — that is correct conservative behavior, not a bug. Run `/api/backfill`
  once (or accumulate ~8 days of live runs) before expecting flags to light up. Deltas
  need ≥2 snapshots on different days.
- **Expiry must match between store and chain.** Deltas/scores only populate for strikes
  whose `(expiry, strike, call_put)` exist in both the built chain and the stored
  history. With live data these share identical real expiry dates; with the synthetic
  MockProvider the seeded expiry must be seeded to the mock's resolved expiry or overlap
  is empty (a synthetic-data artifact only).
- **OneDrive + uvicorn --reload** can miss saves → hard restart.
- **`blpapi`/`xbbg` are Terminal-only** (commented in requirements). Install `blpapi`
  from the Bloomberg package index. Off-Terminal, everything runs on the MockProvider.
- **xbbg returns narwhals-wrapped DataFrames** — handled via a `_to_pandas`/`_pd` helper
  in `bloomberg.py`; don't assume raw pandas from provider calls.
- **Environment note (dev machine):** pandas 3.x, numpy 2.x, blpapi 3.26.x, conda env
  `julien_dev` on the Windows Bloomberg Terminal machine.

---

## 11. Suggested next steps (open / nice-to-have)

- Wire the `trend_window` (20d) signal into the UI as a second badge next to the 5d read
  (engine already computes multi-window; frontend currently surfaces the operational one).
- Add a cron/scheduled daily snapshot write so history accumulates without a manual run.
- Persist per-underlying backfill status so the UI can prompt "seed history" only when needed.
- Optional: expose center-of-gravity / max-pain as vertical markers on the OI chart.
```

---

## 12. Amendment — 16 Jul 2026 (bdh shape fix + rate-space transform)

Two changes since the original document. Both were driven by live-Terminal
findings, so §§5-11 above remain accurate except where noted here.

### 12.1 `bdh_fields` was broken live (backfill 500)

**Symptom.** `POST /api/backfill` returned
`500 Backfill failed: cannot insert date, already exists`.

**Cause.** Not SQLite and not Bloomberg — pandas. `bdh_fields` assumed xbbg 0.x's
shape (DatetimeIndex + MultiIndex `(ticker, field)` columns) and did
`df.index.name = "date"` then `reset_index()`. The installed xbbg is the
**Rust/Arrow-backed v1 line**, which has no index concept, so `date` arrives as
an ordinary **column** next to a RangeIndex. Naming the RangeIndex "date"
collided with the existing column.

**Confirmed shape on the Terminal machine** (via `diagnose_bdh.py`): xbbg v1
returns an **already-tidy long frame**, `['ticker','date','field','value']`,
RangeIndex, flat columns. Not the wide shape originally assumed.

**Second bug hiding behind the first.** Because the frame has flat (not
MultiIndex) columns, the old `else` branch would have fired and stamped every
row with `tickers[0]` — silently writing all ~800 members' history under one
ticker. The crash prevented a data-corruption bug.

**Fix.** New module-level `normalise_bdh(raw, tickers, flds)` in
`data/bloomberg.py` — a pure function (testable without a Terminal, mirroring
the bloomberg.py/chain_builder.py split). It *detects* where dates live rather
than asserting it: DatetimeIndex first, then a `date` column, else raises
explicitly. Handles four shapes: 0.x wide/MultiIndex, v1 tidy-long, `date`
column + flat `ticker|field` columns, and stringified tuple labels.
`_split_ticker_field` parses labels. Covered by `tests/test_xbbg_shapes.py`.

**Ops.** `diagnose_bdh.py` (repo root) dumps the raw/unwrapped/normalised shapes;
run it before the backfill on any new machine or after an xbbg upgrade.
Backfill on `SPX Index` days=90 seeded **25,748 rows** (~32 dates/contract of
~62 trading days — patchy, as expected, and well above `min_context_points=8`).

**`POSITIONING_DB` moved off OneDrive.** SQLite on a sync client risks
corruption. `_DEFAULT_DB` resolves at **module import time**, so the env var must
exist *before* uvicorn starts.

### 12.2 New asset class `RATES_PRICE` — rate-space transform

**Problem.** IMM-style rate futures (SOFR/fed funds/Euribor) quote as
`100 - rate` and their options are struck on that PRICE. Previously
`classify_asset("SFRZ6 Comdty")` fell through to `CMDTY`, forcing the user to
ask "below 96 by December" and translate by hand.

**Where the transform sits (do not move this).** The SABR fit and the
Breeden-Litzenberger extraction run in **price space**, because that is where
the options are struck and where Bloomberg quotes the vols. Only the *finished
density* is mapped. Mapping strikes to rates before fitting would reverse the
skew and distort the smile.

`core/breeden_litzenberger.to_rate_space(rnd, ref=100.0)`. The change of
variable `R = ref - P` is affine with `|dP/dR| = 1`, so:

    q_R(r) = q_P(ref - r)          # density carries across untouched
    F_R(r) = 1 - F_P(ref - r)      # note the complement: a call on the
                                   # price is a put on the rate

Arrays are reversed so the rate grid ascends (`quantile`'s `np.interp` needs a
monotone CDF). `call_prices` / `fitted_vols` are carried index-aligned but
remain **price-space** quantities.

**Detection is deliberately strict.** `is_rate_future()` in `data/bloomberg.py`
requires root + IMM month code + 1-2 digit year + ` Comdty`
(`RATE_FUTURE_ROOTS = SFR, SER, FF, ER, SFI, ED, BA`). A false positive would
mirror a commodity's density about 100, so `CLZ6`/`GCZ6` etc. are explicitly
tested as non-matches. Extend `RATE_FUTURE_ROOTS` for new contracts.

**Touched:** `ASSET_DEFAULTS["RATES_PRICE"]` (beta 0.5, shift 0);
`_MULTIPLIER["RATES_PRICE"] = 2500`; `_grid_bounds` price-space branch
(the generic `0.4*min .. 1.7*max` would span ~38-165 for strikes near 96);
`compute_distribution` (transform + `rate_space`/`forward_price_space`/
`rate_future_ref` response keys, smile x-axis mapped, `strike_price_space`
retained); `compute_positioning` (display-only strike mapping so the OI chart
aligns with the rate-space PDF overlay — **the store stays price space**);
MockProvider presets + smile/strike-grid branches; `frontend/app.js` badge.

**`call_put` is NOT relabelled** in rate space. A call on the price is a put on
the rate; quietly flipping the label would misrepresent the contract. It stays
as the real contract type — `rate_space: true` tells the reader to interpret it.

**Tests:** `tests/test_rate_space.py`. The load-bearing assertion is
`P(rate > r) == P(price < ref - r)`; if that ever fails the dashboard is
answering the opposite question. Also covers involution, pdf area, monotone
CDF, forward/mean/quantile mapping, and that SPX/FEDFUNDS paths are unaffected.

### 12.3 Known caveats added

- **Vol convention is unverified for rate futures.** `chain_builder` does
  `if iv > 3.0: iv = iv/100.0`, which assumes a *lognormal percent* vol. Rates
  desks commonly quote **normal (Bachelier) vols in bp**. If Bloomberg returns
  normal vols for SOFR options, an 80bp vol would be read as 0.80 = 80%
  lognormal and the density would be nonsense. **Check `IVOL_MID` on a known
  SOFR option against OVME before trusting a live rate-future read.** Untested
  against live data — the mock seeds lognormal price vols by construction.
- **`_MULTIPLIER["RATES_PRICE"] = 2500`** is the 3M SOFR convention ($25/bp).
  30-day fed funds (FF) and Euribor differ, so `total_premium_notional` is
  wrong for non-SOFR rate futures until the multiplier is keyed by root.
- **Settlement averaging.** SOFR futures settle to compounded average daily
  SOFR over the 3M reference quarter; FF futures to the monthly average of
  daily EFFR. The density therefore describes an **average rate over a period**,
  not the policy rate on a meeting date. This is *not* a per-meeting hike
  probability — WIRP does that. The dashboard's edge is the full distribution
  (tails, "two or more hikes"), which WIRP handles clumsily; it complements
  WIRP rather than replacing it.
- **`FEDFUNDS` is a mock preset, not a Bloomberg ticker.** Live it fails at
  `spot()` (no PX_LAST). Presets refreshed to the Jul-2026 setting
  (target 3.50-3.75%, EFFR 3.63%); they were seeded at 4.50, ~87bp stale.

### 12.4 Backfill on FUTURES options (`securities is required` 500)

**Symptom.** `POST /api/backfill?underlying=SFRZ6 Comdty` returned
`500 Backfill failed: securities is required for HistoricalDataRequest`.
`SPX Index` worked fine.

**Cause.** `backfill_history_rows` built its `meta` dict purely by parsing
ticker strings. `_parse_opt_ticker` expects the equity/index convention
(`SPXW US 07/17/26 C4300 Index`), which embeds the date. A futures option
(`SFRZ6C 96.00 Comdty`) carries the strike and C/P but **no expiry in the
string at all** — no regex can recover it. Every parse returned None, `meta`
was `{}`, and `bdh()` was called with an empty securities list. There was a
guard for empty `members` but none for empty `meta`, so blpapi's opaque error
surfaced instead of an actionable one.

**Fix.** Ticker parsing stays the fast path; anything that fails now falls back
to `_meta_from_bdp()`, which resolves `OPT_STRIKE_PX` / `OPT_EXPIRE_DT` /
`OPT_PUT_CALL` via BDP (C/P also recoverable from the ticker root,
`SFRZ6C` / `SFRZ6P`). An explicit guard raises before `bdh` if nothing
resolves. Costs one extra BDP call on a one-time backfill.

**Also:** `underlying` is now `.strip()`ed in `compute_distribution`,
`compute_positioning` and `backfill_positioning`. A trailing space from a UI
field would otherwise become part of the store primary key and silently split
one underlying's history in two.

**Note.** `_select_members_by_expiry` still groups by *ticker-parsed* expiry,
so for futures options it falls through to `members[:max_options]` (no expiry
spread). Harmless in practice — `OPT_CHAIN` on `SFRZ6 Comdty` returns options
on that one contract — but it would matter if pointed at a multi-expiry
futures chain.

**Tests:** `tests/test_rate_space.py` — `_FakeFuturesBBG` serves SOFR-style
tickers and BDP fields; asserts bdh is never handed an empty list, metadata
resolves via BDP, the guard raises actionably, and whitespace is stripped.
