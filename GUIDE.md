# Reading the Dashboard — A Plain-English Guide

> **Who this is for.** Anyone opening this dashboard without an options background.
> It explains every chart and number on the screen, and, just as importantly, what
> not to read into them. No maths required. For how the thing is built, see
> `README.md`; for how it is wired, see `HANDOVER.md`.

---

## The one idea everything rests on

An option is a bet on where something ends up by a certain date. Someone selling you
the right to buy SPX at 8,000 in December has to decide what to charge, and that price
reveals what they think the odds are. Cheap means they think it is unlikely. Expensive
means they think it is live.

The market lists dozens of these bets at different levels simultaneously. Line up all
those prices and you can work backwards to the full set of odds the market is
implicitly quoting: not just "will it clear 8,000" but the odds on every level at once.

That is what this dashboard does. It is not forecasting anything. It is reading the
market's own betting board.

---

## The headline number

**Probability of event** is the answer to the question you typed. The odds line next to
it says the same thing the way a bookmaker would.

> **The caveat that matters more than any other on this page.** These are
> *risk-neutral* probabilities, which is not quite the same as real-world odds. People
> pay over the odds for protection, the same way you knowingly overpay for house
> insurance because losing the house is unbearable, not because you think a fire is
> likely. Crash insurance is systematically expensive, so the implied probability of a
> big fall is reliably higher than the historical frequency of big falls.
>
> Read these numbers as **"what the market charges for this bet,"** not "what will
> happen." As a rough steer, downside probabilities on equities are usually overstated
> by this measure.

---

## The header row

| Field | What it means |
|---|---|
| **Underlying** | What you asked about. |
| **Asset class** | How the app is treating it. Controls a few conventions behind the scenes. |
| **Expiry** | The date the bet actually settles. |
| **T (yrs)** | Time to that date in years. 0.42 means about five months. |
| **Forward** | The market's central level. |
| **Data source** | Bloomberg, or Synthetic. |

**Expiry is worth noticing.** The app picks the nearest *listed* expiry to your target
date, so if you asked about December you get whatever December date the exchange
actually lists. It will not be exactly your date.

**Forward** is roughly where you could lock in a price today for that date. It is the
anchor the whole distribution sits around. Not a forecast; more the level at which the
market has no opinion either way.

**Synthetic** means made-up demo numbers, shown when there is no Bloomberg Terminal
attached. Never read anything into a synthetic result.

---

## Risk-neutral density (PDF)

The picture of the market's expectations. The horizontal axis is where the underlying
might end up. The height of the curve says how relatively likely each level is. The
peak is the market's single most-expected landing spot, and the curve tails off towards
levels it considers far-fetched.

The shaded green region is your event.

> **The probability is the AREA of that shading, not the height of the curve.**
> This is the one thing people get wrong looking at this chart. A tall narrow sliver
> can be less likely than a low wide one.

The dashed orange line is your threshold. The dotted blue line is the forward. The gap
between those two lines is basically your question: how far are you asking the market
to travel from where it currently sits?

If the curve is lopsided rather than a symmetrical bell, that is real information. It
means the market sees more room to move one way than the other.

---

## Cumulative distribution (CDF)

The same information as a running total. At any level on the horizontal axis, the
height of the line is the probability of finishing at or below that level. It starts at
zero on the left and climbs to one on the right, because the thing has to end up
somewhere.

Its practical use: you can read any threshold straight off it without retyping a
question. Find your level, read across, and that is your probability of finishing
below. One minus that is above.

The steep middle section is where the market thinks the action is. The flat parts at
either end are the levels it has largely dismissed.

---

## Volatility smile — market vs SABR fit

**A quality check, not an answer.** Ignore it when it looks fine; pay attention when it
does not.

The dots are actual market prices, converted into a common unit. The line is the smooth
curve the app fitted through them.

> **If the line passes close to the dots, the density above is trustworthy. If the dots
> scatter well off the line, do not believe the probability to the second decimal.**

Why "smile"? If markets believed in a simple bell curve, that line would be flat. It is
not. It usually tilts or curls up at the edges, which is the market saying it charges
extra for extreme outcomes because it does not believe in tidy bell curves either. The
shape of that tilt is precisely what makes the density lopsided.

---

## Distribution statistics

**Forward, mean, mode, median** are four different notions of "the middle," and the
gaps between them are the interesting part.

- **Mode** — the single most likely level. The peak of the curve.
- **Median** — the 50/50 point. Half the probability either side.
- **Mean** — the average outcome.

When they disagree, the distribution is lopsided, and the direction of the disagreement
tells you which way.

**Std** is how spread out the whole thing is. Bigger means the market is less sure.

**p05, p25, p75, p95** are percentiles. p05 is the level with only a 5% chance of
finishing below it; p95 has a 5% chance of finishing above.

- **p05 to p95** — roughly the market's 90% confidence range. If someone asks "what is
  the realistic range," this is it.
- **p25 to p75** — the middle half.

---

## SABR fit

The internals of the curve-fitting. Mostly ignorable, with one exception.

**rmse** is fit quality, and it is the number to actually look at. Small means the
smooth line went through the dots nicely. Large means it did not, and everything
upstream is on shaky ground. It is the numerical version of eyeballing the smile chart.

The rest are knobs describing the shape. They do not independently tell you anything
the charts do not:

- **alpha** — roughly the overall level of uncertainty.
- **rho** — the tilt. Negative means downside protection costs more, which is normal
  for equities.
- **nu** — how curved and fat-tailed the shape is.
- **beta** — fixed by asset class rather than fitted.

---

## Where the conviction is

Everything above is derived from **prices**. This section is about **positions**: where
real money is actually sitting.

### Open interest by strike

Counts contracts currently open at each level. A tall bar means a lot of people have
bets there. The density is overlaid so you can compare the two directly, and that is
the point: **does the money sit where the probability is?** When a big pile of
positions sits somewhere the density says is unlikely, someone disagrees with the
consensus, and that is worth a look.

Colours are the app's read on each level:

| Colour | Read |
|---|---|
| Green | High conviction |
| Blue | Moderate |
| Grey | Low |
| Red | Conflicting |
| Dark | Not enough history yet |

### Conviction magnitude

How hard positions are building, and which way. Bars up mean building bets on the level
rising; down means falling. The height is strength, not size.

The reads combine three things: whether positions are growing, whether today's trading
is brisk relative to what is already open, and whether prices are confirming. That last
one carries the most weight, since it is the market pricing the view rather than just
holding it.

> **"Conflicting" is a genuine finding, not a failure.** It appears when the signals
> disagree, for instance when positions pile up but prices do not budge, meaning the
> market is absorbing size without getting more worried. It is flagged rather than
> averaged away precisely because averaging it would produce a bland middling number
> that hides the disagreement.

### Positioning summary

- **Put/call OI ratio** — puts are downside bets, calls upside. Above 1 means more
  downside positioning, though a lot of that is hedging rather than bearishness.
- **Centre of gravity** — the average level of all that positioning, weighted by size.
  Roughly, where the crowd is clustered.
- **Max pain** — the level at which the largest number of options expire worthless.
  **Treat with real scepticism.** The theory that prices get pulled towards it is
  folklore more than established fact, and the evidence is thin. Interesting to note,
  not something to trade off.
- **Premium notional** — the actual cash paid for all those positions. What was
  *risked*, rather than how big the bet is.

### Top conviction strikes

Ranks the levels by strength, with the agreement count shown as n out of 3. A high
magnitude with 3/3 agreement is a much stronger signal than the same magnitude at 2/3.

---

## The honest limitations

**Open interest cannot tell you who is winning.** Every contract has a buyer and a
seller with opposite views. It tells you where activity is concentrated, not which side
is right or what "the market" thinks. It is a map of where the arguments are happening.

**n/a reads are conservative, not broken.** They mean the app does not yet have enough
stored history for that level.

**Risk-neutral is not real-world.** Worth repeating, because it is the one that could
actually mislead you. This tells you what the market charges, and what it charges
reflects both what it expects *and* what it is afraid of. Those are not the same thing.

---

## Quick reference

| If you want to know… | Look at |
|---|---|
| The answer to your question | Probability of event |
| The realistic range | p05 to p95 |
| Whether to trust the answer | Smile chart, and rmse |
| How lopsided the view is | Gap between mean, median and mode |
| Where the money actually is | Open interest by strike |
| Whether the money agrees with the odds | Density overlaid on the OI chart |
| Whether positioning is building or stale | Conviction magnitude |
| The probability of any *other* level | Read it off the CDF |
