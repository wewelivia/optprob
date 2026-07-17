# Plain-English interpreter

A rule-based assessment engine for the option-implied probability dashboard,
in the same spirit as the event parser: no LLM, fully auditable, every
sentence assembled from the computed numbers. For each query it renders an
"Assessment" panel between the probability headline and the charts, with four
sections: what is priced (probability bucket, threshold distance from the
forward in standard deviations, the 90% range, skew), positioning (put/call
OI, OI centre of gravity vs the forward, max pain, top conviction strikes,
with rate-space contracts translated into rate language), the contrarian view
(which side is the fade, its approximate binary odds, and the natural family
of structures per asset class), and caveats (risk-neutral vs real-world,
expiry mapping, fit quality). It never raises: missing positioning history
degrades the text to a pricing-only read.

## New files

    backend/app/core/interpreter.py    the engine (pure, no I/O)
    tests/test_interpreter.py          offline tests (all passing) + a worked
                                       Fed funds example printed at the end

## Changed files (drop over the existing OptProb folder)

    backend/app/main.py                /api/distribution now attaches an
                                       `interpretation` object; positioning is
                                       fetched best-effort from the cached
                                       chain, so added latency is minimal
    backend/app/models/schemas.py      DistributionResponse gains optional
                                       `interpretation`
    frontend/index.html                Assessment panel + scoped styles
    frontend/app.js                    renderInterpretation(); everything else
                                       unchanged

## Notes

The interpreter states probabilities and generic structure families (spreads,
wings, risk reversals); it deliberately never sizes, prices or recommends a
specific ticket, and the caveats section labels the output as an automated
read of market pricing for internal research. Positioning conviction needs
snapshot history: run POST /api/backfill once per underlying to seed it,
otherwise the positioning section says it is reading pricing alone.
