# Serie A Stats Model

A Python research tool for estimating Serie A probabilities and comparing those estimates with public Sisal prices. It does not place bets or establish a profitable betting strategy.

## Setup (macOS)

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements_seriea.txt
    python3 seriea_value_model.py fetch
    python3 seriea_value_model.py backtest --bankroll 60
    python3 seriea_value_model.py sisal --bankroll 60

The Sisal command downloads the official public price-sheet PDF and displays **all available Serie A 1X2 and Under/Over 2.5 odds**, even if the independent forecasting model is not validated. Each fixture is shown with the odds and **Sisal's implied probability after proportional margin removal**. Within each market, the highest implied probability is marked as the most likely outcome *according to the bookmaker*; 1X2 and U/O are ranked separately. The Poisson/Dixon–Coles probability is shown in a separate column for research, not promoted as more accurate.

The terminal prints the full odds board. `seriea_data/sisal_report.md` contains a per-fixture summary and **all outcomes**; `seriea_data/sisal_analisi.csv` contains the ranked rows including original status and validation flags. Unrecognized teams still appear with bookmaker probabilities, but without an independent Poisson forecast. If the results feed is incomplete, odds are still listed and marked, while the model forecast is exploratory.

**All Sisal stakes remain zero until separate validation on timestamped Sisal odds exists.** A Bet365 backtest cannot validate Sisal, and the report does not claim an investment edge.

## Experimental xG + Elo (2026-11)

Three **independent hypotheses** have been implemented and compared with bookmaker
opening probabilities on the **same matches**:

- Elo-only: ratings updated from actual match outcomes, in chronological date batches.
- xG-only: recent, time-weighted expected goals for/against with shrinkage and home advantage.
- xG+Elo: a fixed geometric combination of their expected-goal rates; no bookmaker prices in inputs.

All coefficients are **fixed research hypotheses**, not fitted to make a historical ROI look high.
The program reports log loss and Brier by season and an equal-fixture comparison; it does
not select a winner based on retrospective profit.

xG source: https://www.xgtables.io/understat-data/ republishes historical Understat
match xG through 2025/26. Its 2026/27 figures are Sportmonks
(https://www.xgtables.io/sources/), a DIFFERENT xG model.
Team xG strength calculations therefore use matches from the **same season and provider**.
No mean, total, or relative strength is mixed across sources.

Usage:

    python3 seriea_value_model.py xg-fetch
    python3 seriea_value_model.py xg-backtest --start 2025-08-01
    python3 seriea_value_model.py sisal --bankroll 60

The first command downloads xG CSVs for 2023/24 through 2026/27, checks their
structure, and reconciles each completed match against official scores by date,
home team and away team. Conflicting scores **abort ingestion**; missing xG is
excluded, never converted into zero or replaced with actual goals.

The second command writes:

- xg_elo_predictions.csv: every evaluated method and outcome, including errors.
- xg_elo_metrics.csv: per-season accuracy metrics and sample sizes.
- xg_elo_report.md: *same-match* comparison versus Bet365 opening prices.

The Sisal report adds Elo and xG+Elo probabilities and a table of their
**differences from Sisal**, not betting suggestions. If current-season xG coverage
is inadequate, xG+Elo is shown as n/d and Elo remains visible. Data from the
current match are NEVER used to predict it, even if a CSV was downloaded later.

**Important:** the historical 2025/26 season was used during previous model
development. Its new results are **retrospective** and must not be called a fresh
untouched holdout. Independent prospective evaluation is required before any
claim of predictive skill. A discrepancy in Sisal quotes does not establish value.

## Market comparison: bookmaker consensus, not just lowest odds

The next experiment compares market prices rather than blindly ranking the
lowest odds. It does **not** infer a profitable strategy from bookmaker agreement.

### Historical market quality

    python3 seriea_value_model.py market-backtest --start 2025-08-01

For each completed Serie A fixture, reads the **individual bookmaker opening
1X2 quotes** from the existing Football-Data files, excluding:
- market best/average aggregates, which are not independent bookmakers;
- closing odds, which were not available at opening;
- Pinnacle/PS since **2025-07-23**, because Football-Data itself flags
  systematically outdated public Pinnacle API odds from that date;
- incomplete/invalid bookmaker price sets.

Each available bookmaker is compared against the **median no-vig probabilities
from at least three OTHER bookmakers**, excluding the target's own price to
avoid self-comparison. The resulting price-premium indicator does **not** imply
positive real expected value; bookmakers are not a ground-truth probability source.

The independent market consensus and Bet365's original no-vig opening
probabilities are scored on **identical completed fixtures** with log loss and
Brier. This historical comparison is retrospective, not an untouched holdout;
it includes negative examples. A bootstrapped day-cluster interval measures
the difference in log loss rather than pretending to measure betting profit.

Files: `market_historical_report.md`, `market_historical_prices.csv`,
`market_historical_scores.csv` in `seriea_data/`.

### Observed live snapshots, entered from actual sources

    python3 seriea_value_model.py market-compare --quotes /path/to/quotes.csv

A blank schema is provided in `market_quotes_template.csv`. Required fields:

`captured_at,kickoff,home,away,bookmaker,market_group,outcome,odds`

Each **real price** must be collected BEFORE kickoff and include an ISO-8601
timezone offset. One bookmaker's 1X2 requires all of 1/X/2; U/O 2.5 requires
both `OVER2.5` and `UNDER2.5`. All sources within a fixture and market
must have capture timestamps within **120 seconds** of each other, or the group
is rejected. Four complete bookmaker records are needed for each independent
leave-one-out comparison.

No example or fake market prices are supplied: this command accepts genuine
observations only. The software cannot attest that prices transcribed by the
user were genuinely available or executable, and it does not automatically
scrape additional bookmaker websites.

Output: `market_live_comparison.csv`, `market_live_report.md`. The
price premium and adverse-market scenario are comparisons, **not** statistically
validated edges or betting recommendations. The adverse scenario is NOT a
confidence interval. Simultaneous book price availability, liquidity,
limits and execution remain unverified.

The Sisal PDF importer is still available separately; it does not invent
prices from other bookmaker feeds. Automatic multi-source live comparisons
will require a legitimate, timestamped odds provider.

## Statistical evaluation

The model now compares four distinct sets of probabilities:

1. Independent Poisson (home advantage, attack/defence, time decay and L2 regularisation).
2. A conditional, two-stage Dixon–Coles low-score correction. It is not a joint DC maximum-likelihood fit.
3. Bet365 opening prices with the bookmaker margin removed proportionally.
4. Linear pooling of the DC model and the market. Pool weights are selected on the earlier season **using log loss only**, never by maximizing retrospective ROI.

**Chronological protocol:** calibration on 2024-09-01 through 2025-07-31, then a predeclared holdout beginning 2025-08-01. On every date, model fitting uses only older match results. Candidate weight values are fixed in advance at 0, 0.25, 0.50, 0.75 and 1. If calibration history is insufficient, the market-only fallback is used.

**Strict prices:** backtests use historical Bet365 opening quotes (B365H/B365D/B365A and B365>2.5/B365<2.5). No silent switching to closing odds. A historical opening quote still does not guarantee that a wager was available at that price.

The earlier minus-4.6%-ROI test used a different historical price selection method: its results must not be compared directly with the new ROI.

## Reports

Generated under the seriea_data directory:

- validation_report.md: model versus market, log loss, Brier, paper-bet ROI, drawdown, and diagnostics by bet type and odds band.
- validation_profile.json: fixed calibration weights, holdout metrics, and Sisal validation status.
- backtest_predictions.csv: all evaluated probabilities and realised outcomes, **not only winning selections**.
- backtest_bets.csv: all hypothetical wagers, including losses.
- backtest_segments.csv: segmented results by market and odds range.
- calibration_bins.csv: predicted versus observed frequencies by probability bucket.
- sisal_analisi.csv: **complete odds board**, rank within 1X2 and Under/Over, bookmaker implied probability no-vig and independent Poisson/Dixon–Coles probability in different columns.
- sisal_report.md: all quoted outcomes plus a compact per-fixture table of the highest-probability outcome in each market (based on Sisal no-vig).
- sisal_forecast_archive.csv: timestamped snapshots of predictions and quotes **observed before kickoff**.
- sisal_audit_matches.csv: later scoring of completed Sisal fixtures already present in the snapshot archive.
- risultati_integrati_open_matchday.csv: any missing results cross-checked against the secondary daily schedule.

Use the audit after matches have been completed and the local Football-Data results are available:

    python3 seriea_value_model.py sisal-audit

It scores the **first** archived pre-match snapshot for each fixture/market, not whichever quote looks most profitable after the event. It refuses to extrapolate an advantage from a few results.

Unit tests and reproducible evaluation:

    python3 -m unittest discover -p 'test_*.py' -v
    python3 seriea_value_model.py backtest --start 2025-08-01 --bankroll 60

GitHub Actions runs both on a pull request using the CSV data committed in the repository and uploads all results, including losses.

## Policy and limitations

- Paper-bet evaluation still uses the original, **unoptimised** EV >= 5%, model-edge >= 3 percentage-point thresholds and 1% maximum stake per fixture. The thresholds are not changed to achieve a desired ROI.
- Better forecasting metrics after market anchoring are **not** an independent advantage over a market used as input.
- A Bet365 backtest is not equivalent to a Sisal backtest. Real Sisal quotes must be collected prospectively, with timestamps, to evaluate what was genuinely observable.
- No validated xG, lineups, injuries or in-play information are used or invented.
- Neither positive historical ROI nor a positive bootstrap interval guarantees future results.
- The source for current Serie A results is daily, not live. Incomplete games may prevent the current-odds analysis.
- The original EUR 60 bankroll is a maximum exposure budget, not an investment principal. The program performs no automatic wagering.

## Sources

Football-Data (historical Serie A results and Bet365 prices):
https://www.football-data.co.uk/italym.php

Sisal public price sheet:
https://www.sisal.it/scommesse-matchpoint/foglio-quote

open-matchday supplementary fixture and result checks:
https://github.com/oladeeayo/open-matchday

Dixon and Coles (1997), *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, Journal of the Royal Statistical Society, Series C.

Synthetic tests verify code behaviour, not predictive skill. Only evaluation against authentic, unseen historical and later prospective Sisal data can support claims of reliability. Zero qualifying bets is a valid outcome.
