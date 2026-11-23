#!/usr/bin/env python3
"""Serie A probabilistic model (educational/research use, not profit assurance).

Poisson attack/defence with home advantage, exponential time decay, ridge
regularisation, and strictly chronological out-of-sample backtesting.
Requires numpy, pandas, scipy. Historical CSVs: football-data.co.uk (I1).
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

DATA_DIR = Path(__file__).resolve().parent / 'seriea_data'
SEASONS = ['2324', '2425', '2526', '2627']
GOAL_LIMIT = 12
HALF_LIFE_DAYS = 365.0
RIDGE = 3.0
MIN_HISTORY = 300
MIN_TEAM_MATCHES = 8
MIN_EV = 0.05
MIN_NO_VIG_EDGE = 0.03
KELLY_FRACTION = 0.25
MAX_BANKROLL_FRACTION = 0.01

# Verified Serie A international breaks. End is the FIRST scheduled league
# kickoff (local Europe/Rome time). Do not assume that a long gap is a break:
# new breaks must be checked against the official league calendar.
VERIFIED_BREAKS = (
    (date(2026, 11, 8), datetime(2026, 11, 21, 15, 0)),
)


def history_freshness(last_result: pd.Timestamp, now: datetime) -> tuple[bool, str]:
    """Determine whether history might be missing completed league games.

    A calendar-day-only rule would wrongly reject a valid international break.
    Conversely, after the first league fixture resumes, stale history is a
    genuine data-quality concern, even if the gap began with a break.
    """
    stale_days = (now.date() - last_result.date()).days
    if stale_days <= 9:
        return True, 'Risultati recenti secondo la soglia prudenziale.'
    for final_day, first_kickoff in VERIFIED_BREAKS:
        if last_result.date() == final_day and now < first_kickoff:
            return True, ('Sosta nazionale verificata: nessuna giornata di Serie A '
                          'disputata dopo l’ultimo risultato disponibile.')
    return False, ('Possibili partite di campionato già disputate e assenti dallo storico; '
                   'il modello resta utilizzabile soltanto per un report esplorativo.')


def fetch(seasons: list[str], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        if len(season) != 4 or not season.isdigit():
            raise ValueError(f'Invalid season {season!r}; expected 2425, 2526, etc.')
        url = f'https://www.football-data.co.uk/mmz4281/{season}/I1.csv'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (educational analysis)'})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                raw = response.read()
            preview = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig', nrows=1)
            if not {'Date', 'HomeTeam', 'AwayTeam'}.issubset(preview.columns):
                raise ValueError('The downloaded CSV has unexpected columns')
            path = directory / f'{season}_I1.csv'
            path.write_bytes(raw)
            print(f'OK {url} -> {path} ({len(raw):,} bytes)')
        except Exception as exc:
            print(f'WARNING: unable to download {season}: {exc}', file=sys.stderr)


def read_history(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob('*_I1.csv'))
    if not files:
        raise FileNotFoundError(f'No historical CSV found in {directory}. First run: python seriea_value_model.py fetch')
    parts = []
    for path in files:
        df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        required = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if any(x not in df.columns for x in required):
            print(f'WARNING: skipping incomplete file {path}', file=sys.stderr)
            continue
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce').dt.normalize()
        for col in ['FTHG', 'FTAG']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        parts.append(df)
    if not parts:
        raise ValueError('No usable historical CSV')
    data = pd.concat(parts, ignore_index=True)
    data = data.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
    data = data[(data.FTHG >= 0) & (data.FTAG >= 0)]
    data = data.drop_duplicates(subset=['Date','HomeTeam','AwayTeam'], keep='last')
    return data.sort_values(['Date','HomeTeam']).reset_index(drop=True)


def odds_from_row(row: pd.Series) -> dict[str, float]:
    # Never mix closing and opening prices within the same 1X2 or O/U market.
    def get(col: str) -> float | None:
        value = pd.to_numeric(row.get(col), errors='coerce')
        return float(value) if pd.notna(value) and float(value) > 1.0 else None

    odds: dict[str, float] = {}
    markets = [
        (('1', 'X', '2'),
         [('B365CH', 'B365CD', 'B365CA'), ('B365H', 'B365D', 'B365A')]),
        (('OVER2.5', 'UNDER2.5'),
         [('B365C>2.5', 'B365C<2.5'), ('B365>2.5', 'B365<2.5')]),
    ]
    for labels, versions in markets:
        for columns in versions:
            values = [get(col) for col in columns]
            if all(value is not None for value in values):
                odds.update(zip(labels, values))
                break
    return odds


def no_vig(odds: dict[str, float], outcomes: tuple[str,...]) -> dict[str,float]:
    if any(k not in odds for k in outcomes):
        return {}
    inverse = {k: 1.0 / odds[k] for k in outcomes}
    total = sum(inverse.values())
    return {k: v / total for k, v in inverse.items()}


@dataclass
class Model:
    names: list[str]
    attack: np.ndarray
    defense: np.ndarray
    home_base: float
    away_base: float

    def goals(self, home: str, away: str) -> tuple[float,float]:
        lookup = {name: idx for idx,name in enumerate(self.names)}
        ah = float(self.attack[lookup[home]]) if home in lookup else 0.0
        da = float(self.defense[lookup[away]]) if away in lookup else 0.0
        aa = float(self.attack[lookup[away]]) if away in lookup else 0.0
        dh = float(self.defense[lookup[home]]) if home in lookup else 0.0
        return float(np.exp(self.home_base + ah - da)), float(np.exp(self.away_base + aa - dh))

    def probabilities(self, home: str, away: str) -> dict[str,float]:
        lh, la = self.goals(home, away)
        axis = np.arange(GOAL_LIMIT + 1)
        scores = np.outer(poisson.pmf(axis, lh), poisson.pmf(axis, la))
        scores /= scores.sum()  # Renormalise negligible truncated tails.
        gh, ga = np.indices(scores.shape)
        return {
            '1': float(scores[gh > ga].sum()),
            'X': float(scores[gh == ga].sum()),
            '2': float(scores[gh < ga].sum()),
            'OVER2.5': float(scores[gh + ga >= 3].sum()),
            'UNDER2.5': float(scores[gh + ga <= 2].sum()),
        }


def fit(history: pd.DataFrame, as_of: pd.Timestamp) -> Model:
    history = history[history.Date < as_of]
    if len(history) < MIN_HISTORY:
        raise ValueError(f'Insufficient historical results: {len(history)} < {MIN_HISTORY}')
    names = sorted(set(history.HomeTeam).union(history.AwayTeam))
    positions = {name: i for i,name in enumerate(names)}
    n = len(names)
    home_ix = history.HomeTeam.map(positions).to_numpy(dtype=int)
    away_ix = history.AwayTeam.map(positions).to_numpy(dtype=int)
    goals_h = history.FTHG.to_numpy(dtype=float)
    goals_a = history.FTAG.to_numpy(dtype=float)
    ages = (as_of - history.Date).dt.days.to_numpy(dtype=float)
    weight = np.exp(-math.log(2.0) * ages / HALF_LIFE_DAYS)
    weight = weight / max(float(weight.mean()), 1e-8)

    def objective(x: np.ndarray) -> tuple[float,np.ndarray]:
        attack = x[:n] - x[:n].mean()
        defense = x[n:2*n] - x[n:2*n].mean()
        h_log = np.clip(x[-2] + attack[home_ix] - defense[away_ix], -4.0, 4.0)
        a_log = np.clip(x[-1] + attack[away_ix] - defense[home_ix], -4.0, 4.0)
        mh = np.exp(h_log)
        ma = np.exp(a_log)
        loss = np.sum(weight*(mh-goals_h*h_log+gammaln(goals_h+1)+
                              ma-goals_a*a_log+gammaln(goals_a+1)))
        loss += RIDGE * (attack@attack + defense@defense)

        eh = weight * (mh - goals_h)
        ea = weight * (ma - goals_a)
        ga = np.bincount(home_ix, weights=eh, minlength=n) + np.bincount(away_ix, weights=ea, minlength=n)
        gd = -np.bincount(away_ix, weights=eh, minlength=n) - np.bincount(home_ix, weights=ea, minlength=n)
        ga += 2*RIDGE*attack
        gd += 2*RIDGE*defense
        grad = np.concatenate([ga-ga.mean(), gd-gd.mean(), [eh.sum(), ea.sum()]])
        return float(loss), grad

    x0 = np.zeros(2*n+2)
    x0[-2] = math.log(max(float(np.average(goals_h, weights=weight)), 0.1))
    x0[-1] = math.log(max(float(np.average(goals_a, weights=weight)), 0.1))
    solution = minimize(objective, x0, jac=True, method='L-BFGS-B',
                        options={'maxiter':250, 'ftol':1e-10})
    if not solution.success and np.linalg.norm(solution.jac) > 0.1:
        raise RuntimeError(f'Optimizer did not converge: {solution.message}')
    x = solution.x
    return Model(names, x[:n]-x[:n].mean(), x[n:2*n]-x[n:2*n].mean(),
                 float(x[-2]), float(x[-1]))


def qualify(probs: dict[str,float], odds: dict[str,float]) -> list[dict]:
    benchmarks = {}
    for keys in [('1','X','2'), ('OVER2.5','UNDER2.5')]:
        benchmarks.update(no_vig(odds, keys))
    output = []
    for name, p in probs.items():
        if name not in odds or name not in benchmarks:
            continue
        price = odds[name]
        ev = p*price - 1.0
        edge = p-benchmarks[name]
        if ev < MIN_EV or edge < MIN_NO_VIG_EDGE:
            continue
        full_kelly = max(0.0, (p*price-1.0)/(price-1.0))
        frac = min(MAX_BANKROLL_FRACTION, KELLY_FRACTION*full_kelly)
        output.append({'market':name, 'prob':p, 'odds':price,
                       'fair_odds':1/p, 'ev':ev, 'edge':edge, 'fraction':frac})
    return sorted(output, key=lambda x: x['ev'], reverse=True)


def actual_winner(row: pd.Series, market: str) -> bool:
    hg, ag = int(row.FTHG), int(row.FTAG)
    return {'1': hg>ag, 'X': hg==ag, '2': hg<ag,
            'OVER2.5': hg+ag>2, 'UNDER2.5': hg+ag<=2}[market]


def strength_counts(history: pd.DataFrame, home: str, away: str) -> tuple[int,int]:
    return (int(((history.HomeTeam==home)|(history.AwayTeam==home)).sum()),
            int(((history.HomeTeam==away)|(history.AwayTeam==away)).sum()))


def backtest(df: pd.DataFrame, start: pd.Timestamp, bankroll: float) -> None:
    days = sorted(df[df.Date >= start].Date.unique())
    predictions = []
    decisions = []
    capital = bankroll
    peak = bankroll
    drawdown = 0.0
    fit_count = 0
    for date in days:
        date = pd.Timestamp(date)
        history = df[df.Date < date]
        if len(history) < MIN_HISTORY:
            continue
        model = fit(history, date)
        fit_count += 1
        for _, row in df[df.Date == date].iterrows():
            odds = odds_from_row(row)
            reference = no_vig(odds, ('1','X','2'))
            if not reference:
                continue
            p = model.probabilities(str(row.HomeTeam), str(row.AwayTeam))
            outcome = '1' if row.FTHG>row.FTAG else ('X' if row.FTHG==row.FTAG else '2')
            vector = np.array([p[k] for k in ('1','X','2')])
            market = np.array([reference[k] for k in ('1','X','2')])
            truth = np.array([float(k==outcome) for k in ('1','X','2')])
            predictions.append({
                'date':date, 'home':row.HomeTeam, 'away':row.AwayTeam,
                'model_ll': -math.log(max(p[outcome],1e-9)),
                'market_ll': -math.log(max(reference[outcome],1e-9)),
                'model_brier': float(np.square(vector-truth).sum()),
                'market_brier': float(np.square(market-truth).sum()),
            })
            n_home,n_away = strength_counts(history,str(row.HomeTeam),str(row.AwayTeam))
            if min(n_home,n_away) < MIN_TEAM_MATCHES:
                continue
            choices = qualify(p,odds)
            if not choices or capital<=0:
                continue
            selected = choices[0]  # only one, to avoid correlated selections per fixture
            stake = capital * selected['fraction']
            if stake < 0.001:
                continue
            won = actual_winner(row,selected['market'])
            profit = stake*(selected['odds']-1) if won else -stake
            capital += profit
            peak = max(peak, capital)
            drawdown = max(drawdown, (peak-capital)/peak)
            decisions.append({'date':str(date.date()),'home':row.HomeTeam,'away':row.AwayTeam,
                              'market':selected['market'], 'prob':selected['prob'],
                              'odds':selected['odds'], 'ev':selected['ev'],
                              'stake':stake,'won':won,'profit':profit,'capital':capital,
                              'unit_return': (selected['odds']-1) if won else -1.0})
    if not predictions:
        print('No evaluable matches. Check start date and available historical seasons.')
        return
    m = pd.DataFrame(predictions)
    print(f'\nOUT-OF-SAMPLE from {start.date()} | {len(m)} scored matches | {fit_count} daily model fits')
    print(f'Mean log loss (lower is better): model {m.model_ll.mean():.4f}, market {m.market_ll.mean():.4f}')
    print(f'Mean Brier (lower is better): model {m.model_brier.mean():.4f}, market {m.market_brier.mean():.4f}')
    print(f'Qualification rules: EV >= {MIN_EV:.0%}, model probability >= no-vig market + {MIN_NO_VIG_EDGE:.0%}, at least {MIN_TEAM_MATCHES} matches/team')
    if not decisions:
        print('NO BETS qualified (the correct outcome may be zero bets).')
        return
    bets = pd.DataFrame(decisions)
    staked = bets.stake.sum()
    profit = bets.profit.sum()
    print(f'Candidate bets: {len(bets)} | win rate: {bets.won.mean():.1%} | total stakes: €{staked:.2f}')
    print(f'Net result: €{profit:+.2f} | ROI on stakes: {profit/staked:+.1%} | final simulated bankroll: €{capital:.2f}')
    print(f'Maximum drawdown: {drawdown:.1%}')
    # Non-parametric bootstrap of equal-stake unit returns (descriptive, not certainty).
    if len(bets)>=20:
        rng=np.random.default_rng(2026)
        rr=bets.unit_return.to_numpy()
        ci=np.quantile(rng.choice(rr,size=(3000,len(rr)),replace=True).mean(axis=1),[0.025,0.975])
        print(f'95% bootstrap interval for equal-stake ROI (approx.): [{ci[0]:+.1%}, {ci[1]:+.1%}]')
    output=DATA_DIR/'backtest_bets.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    bets.to_csv(output,index=False)
    print(f'Full log: {output}')
    print('Reminder: retrospective closing odds may not have been accessible at that price; past ROI is not a forecast.')


def predict(df: pd.DataFrame, fixtures_path: Path, as_of: pd.Timestamp, bankroll: float) -> None:
    fixtures=pd.read_csv(fixtures_path)
    required={'Date','HomeTeam','AwayTeam','OddsH','OddsD','OddsA','OddsOver25','OddsUnder25'}
    if not required.issubset(fixtures.columns):
        raise ValueError('Fixtures CSV requires: '+', '.join(sorted(required)))
    fixtures.Date=pd.to_datetime(fixtures.Date,format='mixed',dayfirst=True,errors='coerce').dt.normalize()
    model=fit(df[df.Date<as_of], as_of)
    history=df[df.Date<as_of]
    print(f'Training: {len(history)} historical matches strictly before {as_of.date()}')
    if len(history) and (as_of - history.Date.max()).days > 9:
        print(f'WARNING: most recent result is {history.Date.max().date()}; source may be stale. Verify completeness before betting.')
    for _, row in fixtures.iterrows():
        if pd.isna(row.Date) or row.Date<as_of:
            print('SKIPPED: fixture date invalid or before analysis cutoff')
            continue
        home,away=str(row.HomeTeam).strip(),str(row.AwayTeam).strip()
        n_home,n_away=strength_counts(history,home,away)
        probs=model.probabilities(home,away)
        odds={k:float(row[col]) for k,col in [('1','OddsH'),('X','OddsD'),('2','OddsA'),
                ('OVER2.5','OddsOver25'),('UNDER2.5','OddsUnder25')]
                if pd.notna(row[col]) and float(row[col])>1.0}
        choices=qualify(probs,odds)
        lh,la=model.goals(home,away)
        print(f'\n{row.Date.date()}  {home} - {away} | expected goals {lh:.2f} : {la:.2f}')
        print(f"Probabilities: 1 {probs['1']:.1%}, X {probs['X']:.1%}, 2 {probs['2']:.1%}, "
              f"O2.5 {probs['OVER2.5']:.1%}, U2.5 {probs['UNDER2.5']:.1%}")
        print('Fair odds: '+', '.join(f'{k} {1/p:.2f}' for k,p in probs.items()))
        print(f'Historical sample by team: {n_home} / {n_away} matches')
        if min(n_home,n_away)<MIN_TEAM_MATCHES:
            print('NO BET: insufficient team history / new promotion')
        elif choices:
            best=choices[0]
            print(f"CANDIDATE (not assured): {best['market']} @ {best['odds']:.2f} | "
                  f"prob {best['prob']:.1%}, EV {best['ev']:+.1%}, "
                  f"vs no-vig {best['edge']:+.1%}, 1/4 Kelly capped stake €{bankroll*best['fraction']:.2f}")
        else:
            print('NO BET: no selection meets both EV and probability-edge gates')


# Names on Sisal's PDF are not always identical to football-data.co.uk.
SISAL_TEAM_ALIASES = {
    'internazionale': 'Inter', 'inter milano': 'Inter', 'ac milan': 'Milan',
    'as roma': 'Roma', 'ssc napoli': 'Napoli', 'hellas verona': 'Verona',
    'verona': 'Verona', 'ac monza': 'Monza', 'ac fiorentina': 'Fiorentina',
    'us lecce': 'Lecce', 'fc torino': 'Torino', 'fc como': 'Como',
    'como 1907': 'Como', 'us sassuolo': 'Sassuolo',
}


def normalized_team(name: str, known: set[str]) -> str | None:
    import unicodedata
    def normalize(value: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFKD',value.casefold())
                       if not unicodedata.combining(c)).replace('.', '').strip()
    exact = {normalize(x):x for x in sorted(known)}
    key=normalize(name)
    if key in exact:
        return exact[key]
    alias=SISAL_TEAM_ALIASES.get(key)
    return exact.get(normalize(alias)) if alias else None


def sisal_analysis(pdf_path: Path | None, data_dir: Path, bankroll: float,
                   *, no_refresh: bool = False,
                   analysis_time: datetime | None = None) -> None:
    """One-command public Sisal quote import + research report; never bets."""
    from sisal_import import (SisalImportError, parse_sisal_pdf,
                              check_freshness, download_sisal_pdf, OFFICIAL_PDF_URL)
    from datetime import date

    now=analysis_time or datetime.now(ZoneInfo('Europe/Rome')).replace(tzinfo=None)
    if pdf_path is None:
        pdf_path=data_dir/'sisal_foglio_quote.pdf'
        payload=download_sisal_pdf(pdf_path)
        print(f'PDF ufficiale scaricato: {pdf_path}')
    else:
        payload=pdf_path.read_bytes()
        print(f'PDF caricato: {pdf_path}')
    fixtures, updated = parse_sisal_pdf(payload, reference_date=now.date())
    check_freshness(updated, now)
    # Pull historical results automatically. Once downloaded, older seasons can
    # be cached locally; the current season is re-fetched on every run.
    if not no_refresh:
        needed=[season for season in SEASONS if not (data_dir/f'{season}_I1.csv').exists()]
        latest_season=SEASONS[-1]
        if latest_season not in needed:
            needed.append(latest_season)
        fetch(needed,data_dir)
    df=read_history(data_dir)
    sync_error = None
    sync_reason = None
    if not no_refresh:
        from results_sync import sync_results, ResultSyncError
        try:
            df, sync_reason = sync_results(df, now, data_dir)
            print(f'RISULTATI INTEGRATI: {sync_reason}')
        except ResultSyncError as exc:
            sync_error = str(exc)
            print(f'ATTENZIONE: integrazione risultati non verificata: {exc}', file=sys.stderr)
    as_of=pd.Timestamp(now.date())
    history=df[df.Date < as_of]
    if len(history)<MIN_HISTORY:
        raise ValueError(f'Risultati insufficienti per il modello: {len(history)} (<{MIN_HISTORY}).')
    last_result=history.Date.max()
    stale_days=(as_of-last_result).days
    if no_refresh:
        history_is_fresh, freshness_reason = history_freshness(last_result, now)
    else:
        history_is_fresh = sync_error is None and sync_reason is not None
        freshness_reason = sync_reason if history_is_fresh else (
            'Controllo completo non riuscito: ' + (sync_error or 'fonte secondaria indisponibile'))
    history_stale=not history_is_fresh
    if history_is_fresh and stale_days > 9:
        print(f'STO RICONOSCENDO UNA SOSTA: {freshness_reason}')
    if history_stale:
        print(
            f'ATTENZIONE: ultimo risultato verificabile al {last_result.date()} '
            f'({stale_days} giorni fa). Creo un report ESPLORATIVO, '
            'senza candidate e senza puntate. Aggiorna lo storico prima di utilizzarlo.',
            file=sys.stderr,
        )
    model=fit(history,as_of)
    known=set(model.names)
    rows=[]
    unrecognized=[]
    skipped_past=0
    for event in fixtures:
        kickoff=datetime.fromisoformat(event['Date']+'T'+event['Time'])
        if kickoff <= now:
            skipped_past += 1
            continue
        home=normalized_team(event['HomeTeamSisal'],known)
        away=normalized_team(event['AwayTeamSisal'],known)
        if home is None or away is None:
            unrecognized.append(f"{event['HomeTeamSisal']} - {event['AwayTeamSisal']}")
            continue
        n1,n2=strength_counts(history,home,away)
        probs=model.probabilities(home,away)
        odds={'1':event['OddsH'],'X':event['OddsD'],'2':event['OddsA']}
        if event['OddsOver25'] is not None and event['OddsUnder25'] is not None:
            odds.update({'OVER2.5':event['OddsOver25'],'UNDER2.5':event['OddsUnder25']})
        no_vig_probs={}
        for outcomes in (('1','X','2'),('OVER2.5','UNDER2.5')):
            no_vig_probs.update(no_vig(odds,outcomes))
        candidates=qualify(probs,odds) if not history_stale and min(n1,n2)>=MIN_TEAM_MATCHES else []
        candidate_markets={x['market']:x for x in candidates}
        for market,price in odds.items():
            p=probs[market]
            ev=p*price-1
            edge=p-no_vig_probs[market]
            bet = candidate_markets.get(market)
            rows.append({
                'Data':event['Date'], 'Ora':event['Time'],
                'Partita':f'{event["HomeTeamSisal"]} - {event["AwayTeamSisal"]}',
                'Mercato':market, 'QuotaSisal':round(price,3),
                'ProbModello':round(p,5), 'QuotaEqua':round(1/p,3),
                'ProbMercatoNoVig':round(no_vig_probs[market],5),
                'EV':round(ev,5), 'Edge':round(edge,5),
                'CampioneCasa':n1,'CampioneOspite':n2,
                'Stato':('STORICO_DA_AGGIORNARE' if history_stale else
                         'CANDIDATA' if bet else
                         'STORICO_INSUFFICIENTE' if min(n1,n2)<MIN_TEAM_MATCHES else 'NO_BET'),
                'PuntataTeorica':round(bankroll*bet['fraction'],4) if bet else 0.0,
                'AggiornamentoSisal':event['SisalUpdated'],
                'UltimoRisultatoStorico':str(last_result.date()),
                'QualitaStorico':'NON_AGGIORNATO' if history_stale else 'OK',
            })
    if not rows:
        raise ValueError('Non ci sono partite future riconosciute da analizzare. '
                         'Controlla data del PDF, nomi delle squadre e palinsesto.')
    report=pd.DataFrame(rows)
    # For report presentation only, allow at most one selection per fixture.
    chosen=set()
    exposure=0.0
    for idx in report.sort_values('EV',ascending=False).index:
        row=report.loc[idx]
        if row.Stato != 'CANDIDATA':
            continue
        key=(row.Data,row.Ora,row.Partita)
        if key in chosen:
            report.loc[idx,'Stato']='NO_BET_CORRELAZIONE'
            report.loc[idx,'PuntataTeorica']=0.0
            continue
        chosen.add(key)
        stake=min(float(row.PuntataTeorica), max(0,bankroll*.05-exposure))
        if stake < .01:
            report.loc[idx,'Stato']='NO_BET_LIMITE_ESPOSIZIONE'
            report.loc[idx,'PuntataTeorica']=0.0
        else:
            report.loc[idx,'PuntataTeorica']=round(stake,2)
            exposure+=stake
    report=report.sort_values(['Stato','EV'],ascending=[True,False])
    data_dir.mkdir(exist_ok=True,parents=True)
    original=pd.DataFrame(fixtures)
    quote_csv=data_dir/'sisal_quote_importate.csv'
    result_csv=data_dir/'sisal_analisi.csv'
    result_md=data_dir/'sisal_report.md'
    original.to_csv(quote_csv,index=False)
    report.to_csv(result_csv,index=False)
    shortlisted=report[report.Stato=='CANDIDATA'].sort_values('EV',ascending=False)
    fmt=lambda x:f'{x:.1%}'
    with result_md.open('w',encoding='utf-8') as f:
        f.write(f'# Analisi Serie A / Sisal\n\nPDF aggiornato: {updated:%d/%m/%Y %H:%M} '
                f'(ora italiana). Analisi: {now:%d/%m/%Y %H:%M}.\n\n')
        f.write(f'Fonte PDF: {OFFICIAL_PDF_URL}\n\n')
        f.write(f'Ultimo risultato utilizzato: {last_result.date()} '
                f'({stale_days} giorni prima dell’analisi).\n\n')
        f.write(f'Controllo calendario: {freshness_reason}\n\n')
        if history_stale:
            f.write('**STORICO DA AGGIORNARE — REPORT ESPLORATIVO.** '
                    'Le partite più recenti possono mancare: tutti i mercati hanno '
                    'stato `STORICO_DA_AGGIORNARE` e **puntata teorica 0 EUR**. '
                    'Non interpretare i valori EV come occasioni sfruttabili.\n\n')
        f.write(f'Capitale ipotetico: {bankroll:.2f} EUR. '
                'Max 1% a puntata e 5% totale esposto; mai puntata automatica.\n\n')
        f.write('| Data | Partita | Esito | Quota | Prob. | EV | Puntata |\n')
        f.write('|---|---|---|---:|---:|---:|---:|\n')
        if shortlisted.empty:
            f.write('| — | Nessuna scommessa supera i filtri | — | — | — | — | — |\n')
        else:
            for _,r in shortlisted.iterrows():
                f.write(f'| {r.Data} {r.Ora} | {r.Partita} | {r.Mercato} | '
                        f'{r.QuotaSisal:.2f} | {fmt(r.ProbModello)} | '
                        f'{fmt(r.EV)} | {r.PuntataTeorica:.2f} EUR |\n')
        f.write('\n**Attenzione:** le quote possono cambiare; il modello non è '
                'validato su storico Sisal. EV stimato non significa profitto garantito. '
                'Non aumentare la puntata al minimo richiesto dal bookmaker.\n')
        if unrecognized:
            f.write('\nPartite non elaborate per squadre sconosciute: '
                    +'; '.join(sorted(set(unrecognized)))+'\n')
    print(f'\nSISAL: aggiornato {updated:%d/%m/%Y %H:%M}; partite PDF {len(fixtures)}; '
          f'partite iniziate {skipped_past}; mercati analizzati {len(report)}')
    if unrecognized:
        print('ATTENZIONE squadre non riconosciute (nessuna previsione): '+ '; '.join(unrecognized))
    if history_stale:
        print('NESSUNA PUNTATA PROPOSTA: storico non aggiornato, report solo esplorativo.')
    elif shortlisted.empty:
        print('NESSUNA CANDIDATA. Non è necessario fare scommesse.')
    else:
        print('\nCANDIDATE TEORICHE (non validate contro Sisal):')
        for _,r in shortlisted.iterrows():
            print(f'{r.Data} {r.Ora} | {r.Partita:<32} {r.Mercato:<9} '
                  f'@ {r.QuotaSisal:.2f} | P {r.ProbModello:.1%} | '
                  f'EV {r.EV:+.1%} | puntata {r.PuntataTeorica:.2f} EUR')
    print(f'\nQuote: {quote_csv}\nAnalisi completa: {result_csv}\nReport leggibile: {result_md}')
    print('IMPORTANTE: validare il modello fuori campione prima di usare denaro reale.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('fetch',help='Download historical Serie A CSV seasons')
    s.add_argument('--seasons',nargs='+',default=SEASONS)
    s=sub.add_parser('backtest',help='Walk-forward evaluation, no future leakage')
    s.add_argument('--start',default='2025-08-01')
    s.add_argument('--bankroll',type=float,default=60.0)
    s=sub.add_parser('predict',help='Predict user-entered fixtures and current odds')
    s.add_argument('--fixtures',type=Path,required=True)
    s.add_argument('--as-of',default=None,help='YYYY-MM-DD; defaults to local today')
    s.add_argument('--bankroll',type=float,default=60.0)
    s=sub.add_parser('sisal',help='Import Sisal PDF and analyze future Serie A automatically')
    s.add_argument('--pdf',type=Path,default=None,help='Use an already downloaded official PDF')
    s.add_argument('--bankroll',type=float,default=60.0)
    s.add_argument('--no-refresh',action='store_true',help='Offline tests only: use cached historical results')
    args=parser.parse_args()
    if args.command=='fetch':
        fetch(args.seasons,args.data_dir)
        return
    if args.bankroll<=0: raise ValueError('Bankroll must be > 0')
    if args.command=='sisal':
        sisal_analysis(args.pdf,args.data_dir,args.bankroll,no_refresh=args.no_refresh)
        return
    df=read_history(args.data_dir)
    if args.command=='backtest':
        backtest(df,pd.Timestamp(args.start),args.bankroll)
    else:
        as_of=pd.Timestamp(args.as_of).normalize() if args.as_of else pd.Timestamp.today().normalize()
        predict(df,args.fixtures,as_of,args.bankroll)

if __name__=='__main__':
    main()
