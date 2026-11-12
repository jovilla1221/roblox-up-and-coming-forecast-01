# Roblox Up-and-Coming Forecast

A make.fun entry for the `roblox-up-and-coming-forecast-01` bounty:
"Predict Roblox's Up-and-Coming top 10."

## What it does

`main.ts` reads the chart history in the input directory (`history.ndjson`
and `task.json`) and writes one `forecast.json`: for each of the chart's
top 10 places, a probability for each named game being at that place over
the 72 hours after entries close, plus a top-10 guess in order.

Because that one forecast is scored against every hourly read of the
window, the program forecasts occupancy frequency, not a single snapshot.
Per game it estimates a latent expected log-rank from a time-weighted
average of its ranks, adds a damped momentum drift (rank velocity plus a
small playing-count trend correction and a Top Trending bonus), then runs
a seeded Monte Carlo over 36 two-hour horizon buckets: each draw ranks all
candidate games with fat-tailed noise that widens with the square root of
the horizon, and the share of draws in which a game holds a place becomes
its probability there. Games with a known `created` date leave the
simulation after they age off the chart (created + 28 days). A share of
every place is deliberately left unnamed for new and surprise entrants,
and no game is given more than a capped probability at any place.

## Model notes

The model was reviewed twice by an external reasoning model (Gemini
"Argon", high reasoning), and every suggestion was kept only where a
truncated-history backtest with the competition's exact scorer scored it
better, or where a safety argument applied to data the practice archive
does not contain. Kept: a playing-residual z-score (audience large for its
rank), a top-3 burnout drift, a time-varying unnamed share, a breakout lag
correction pulling a climbing game's latent rank toward its closing rank,
redistribution of above-cap probability mass to the named games below the
cap, trapezoidal time weights and an offset-aware noise floor (sampling
density invariance for the final's hourly history), plus final-only
guards that stay inert on the practice archive: a hard-capped
one-directional vote-ratio penalty, a dense-regime 12h half-life for the
playing slope, and validity guards on the created-based age-out.
Rejected after testing: shrunk-momentum parameter packages, rank-tiered
noise, higher row cutoffs, and an empirical transition-matrix prior.

A third review round, given the verification results and per-position
diagnostics, raised two technical vetoes that testing confirmed: the
rank/playing regressions now run from two observations up with a ridge
term on the time-variance denominator (a two-observation game in sparse
history keeps a shrunk slope instead of none; a three-read game in hourly
history cannot produce an explosive slope), and the trapezoidal weight of
the closing read is floored at the median read gap so the freshest state
is never down-weighted by a short final interval. Also kept from that
round: a cross-position flattening mixture for breakout candidates, which
lifts their thin top-place probabilities toward their own top-10 average.
Rejected after testing: a sniper clause widening candidates to rank 90,
a continuous burnout ramp, and kernel smoothing over places 6-10.

A final improvement round pooled the forecast over three half-lives
(5h, 6h, 7.5h) and added a tenure hazard for games that have sat on the
chart since the first read without a strong audience; both scored better
in backtests and are in the shipped program.

Backtest of the shipped program on the practice history (truncation
method, exact scorer): weighted mean -37.43 over all future reads and
-35.93 over the first 72 hours, against the persistence baseline's -48.73
on the same cuts.

The program uses only the Node 22 standard library, touches no network,
and is deterministic: every run on the same history writes exactly the
same bytes.

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
cat out/forecast.json
```

Without the environment variables, the program reads `/input` and writes
to `/output/forecast.json`, which is where the make.fun grader expects it.

## Files

- `main.ts` — the entry program (Node 22, standard library only)
- `LICENSE` — MIT license
- `.makefun/owner.json` — the entrant wallet that owns this entry
