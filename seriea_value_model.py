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
from scipy.optimize import minimize, minimize_scalar
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


def odds_from_row(row: pd.Series, *, price_kind: str = 'opening') -> dict[str, float]:
    """Complete market snapshots, never mixed between opening and closing.

    Opening prices are reproducible benchmarks, but executability is not proven.
    """
    if price_kind not in {'opening', 'closing'}:
        raise ValueError('price_kind must be opening or closing')

    def get(col: str) -> float | None:
        value = pd.to_numeric(row.get(col), errors='coerce')
        return float(value) if pd.notna(value) and np.isfinite(value) and float(value)>1 else None

    if price_kind == 'opening':
        markets = [
            (('1', 'X', '2'), ('B365H', 'B365D', 'B365A')),
            (('OVER2.5', 'UNDER2.5'), ('B365>2.5', 'B365<2.5')),
        ]
    else:
        markets = [
            (('1', 'X', '2'), ('B365CH', 'B365CD', 'B365CA')),
            (('OVER2.5', 'UNDER2.5'), ('B365C>2.5', 'B365C<2.5')),
        ]
    odds: dict[str, float] = {}
    for names, cols in markets:
        values = [get(col) for col in cols]
        if all(value is not None for value in values):
            odds.update(zip(names, values))
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
    rho: float = 0.0  # Dixon-Coles low-score parameter learned from older results.

    def goals(self, home: str, away: str) -> tuple[float,float]:
        lookup = {name: idx for idx,name in enumerate(self.names)}
        ah = float(self.attack[lookup[home]]) if home in lookup else 0.0
        da = float(self.defense[lookup[away]]) if away in lookup else 0.0
        aa = float(self.attack[lookup[away]]) if away in lookup else 0.0
        dh = float(self.defense[lookup[home]]) if home in lookup else 0.0
        return float(np.exp(self.home_base + ah - da)), float(np.exp(self.away_base + aa - dh))

    def probabilities(self, home: str, away: str, *, dc: bool = True) -> dict[str,float]:
        lh, la = self.goals(home, away)
        axis = np.arange(GOAL_LIMIT + 1)
        scores = np.outer(poisson.pmf(axis, lh), poisson.pmf(axis, la))
        # Dixon-Coles correction only for the 0/1 score combinations.
        if dc and self.rho:
            rho = self.rho
            scores[0, 0] *= 1.0 - lh * la * rho
            scores[0, 1] *= 1.0 + lh * rho
            scores[1, 0] *= 1.0 + la * rho
            scores[1, 1] *= 1.0 - rho
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('Invalid Dixon-Coles score distribution')
        if poisson.sf(GOAL_LIMIT, lh) > 0.001 or poisson.sf(GOAL_LIMIT, la) > 0.001:
            raise ValueError('Poisson grid truncates too much probability mass')
        scores /= scores.sum()
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
    fitted = Model(names, x[:n]-x[:n].mean(), x[n:2*n]-x[n:2*n].mean(),
                   float(x[-2]), float(x[-1]))
    # Two-stage *conditional* Dixon-Coles fit, not a joint MLE.
    # Attacking/defensive strengths above are fitted without this correction;
    # low-score rho is then estimated from strictly prior matches.
    h_log = x[-2] + fitted.attack[home_ix] - fitted.defense[away_ix]
    a_log = x[-1] + fitted.attack[away_ix] - fitted.defense[home_ix]
    lh, la = np.exp(h_log), np.exp(a_log)
    low = (goals_h <= 1) & (goals_a <= 1)
    if low.sum() >= 25:
        h, a, gh, ga, w = lh[low], la[low], goals_h[low], goals_a[low], weight[low]
        lower = max(-0.15, -0.95 / max(float(lh.max()), float(la.max())))
        upper = min(0.15, 0.95 / max(float((lh * la).max()), 1.0))
        if lower < upper:
            def dc_objective(rho: float) -> float:
                tau = np.where((gh==0) & (ga==0), 1-h*a*rho,
                      np.where((gh==0) & (ga==1), 1+h*rho,
                      np.where((gh==1) & (ga==0), 1+a*rho, 1-rho)))
                return float(-(w*np.log(tau)).sum() + 20.0*rho*rho)
            result = minimize_scalar(dc_objective, bounds=(lower, upper),
                                     method='bounded', options={'xatol':1e-6})
            if result.success and np.isfinite(result.fun):
                fitted.rho = float(result.x)
    return fitted


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


def backtest(df: pd.DataFrame, start: pd.Timestamp, bankroll: float,
             directory: Path = DATA_DIR) -> None:
    from model_evaluation import run_validation
    run_validation(df, start, bankroll, directory)


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
            print(f"RESEARCH ONLY (not Sisal-validated): {best['market']} @ {best['odds']:.2f} | "
                  f"raw Poisson-DC probability {best['prob']:.1%}, "
                  f"raw EV {best['ev']:+.1%}; stake €0.00")
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
    from sisal_reporting import rank_full_board, terminal_board, write_readable_report
    from xg_elo_model import load_xg, fetch_xg, forecast_asof, XGDataError

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
    from model_evaluation import load_validation_profile, blend_probabilities
    model=fit(history,as_of)
    profile=load_validation_profile(data_dir)
    if not no_refresh:
        try:
            fetch_xg(data_dir, seasons=(2026,))
        except XGDataError as exc:
            print(f'ATTENZIONE: aggiornamento xG 2026/27 non riuscito: {exc}',
                  file=sys.stderr)
    try:
        xg_matches,xg_quality=load_xg(data_dir,df)
        print(f"xG verificati: {xg_quality['matched']}; fonti per stagione separate.")
    except XGDataError as exc:
        print(f'ATTENZIONE: xG non verificati: {exc}; uso solo Elo.',file=sys.stderr)
        xg_matches=pd.DataFrame(columns=['Date','HomeTeam','AwayTeam','XGH','XGA','Provider','Season'])

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
        recognized = home is not None and away is not None
        if not recognized:
            unrecognized.append(f"{event['HomeTeamSisal']} - {event['AwayTeamSisal']}")
            n1 = n2 = 0
            raw_probs = None
        else:
            n1,n2=strength_counts(history,home,away)
            raw_probs=model.probabilities(home,away)
        independent = (forecast_asof(history,xg_matches,as_of,home,away)
                       if recognized else {'status':'SQUADRA_NON_RICONOSCIUTA'})
        odds={'1':event['OddsH'],'X':event['OddsD'],'2':event['OddsA']}
        if event['OddsOver25'] is not None and event['OddsUnder25'] is not None:
            odds.update({'OVER2.5':event['OddsOver25'],'UNDER2.5':event['OddsUnder25']})
        no_vig_probs={}
        for outcomes in (('1','X','2'),('OVER2.5','UNDER2.5')):
            no_vig_probs.update(no_vig(odds,outcomes))
        # A good Bet365 backtest CANNOT certify Sisal odds.
        probs = (blend_probabilities(raw_probs, odds, profile['weights'])
                 if raw_probs is not None and profile is not None else no_vig_probs)
        for market,price in odds.items():
            p=probs[market]
            ev=p*price-1
            edge=p-no_vig_probs[market]
            raw=raw_probs[market] if raw_probs is not None else float('nan')
            pred_elo=(independent.get('elo') or {}).get(market,float('nan'))
            pred_xg=(independent.get('xg') or {}).get(market,float('nan'))
            pred_hybrid=(independent.get('hybrid') or {}).get(market,float('nan'))
            rows.append({
                'Data':event['Date'], 'Ora':event['Time'],
                'Partita':f'{event["HomeTeamSisal"]} - {event["AwayTeamSisal"]}',
                'HomeTeam':home or event['HomeTeamSisal'],
                'AwayTeam':away or event['AwayTeamSisal'],
                'Mercato':market, 'QuotaSisal':round(price,3),
                'ProbModello':round(p,5), 'ProbPoissonDC':round(raw,5),
                'EVPoissonDC':round(raw*price-1,5),
                'ProbElo':round(pred_elo,5),
                'ProbXG':round(pred_xg,5),
                'ProbXGElo':round(pred_hybrid,5),
                'DifferenzaEloVsSisal':round(pred_elo-no_vig_probs[market],5),
                'DifferenzaXGEloVsSisal':round(pred_hybrid-no_vig_probs[market],5),
                'FonteXG':independent.get('provider'),
                'StatoXG':independent.get('status'),
                'QuotaEqua':round(1/p,3),
                'ProbMercatoNoVig':round(no_vig_probs[market],5),
                'EV':round(ev,5), 'Edge':round(edge,5),
                'CampioneCasa':n1,'CampioneOspite':n2,
                'Stato':('STORICO_DA_AGGIORNARE' if history_stale else
                         'SQUADRE_NON_RICONOSCIUTE' if not recognized else
                         'STORICO_INSUFFICIENTE' if min(n1,n2)<MIN_TEAM_MATCHES else
                         'NON_VALIDATO_SISAL'),
                'PuntataTeorica':0.0,
                'AggiornamentoSisal':event['SisalUpdated'],
                'UltimoRisultatoStorico':str(last_result.date()),
                'QualitaStorico':'NON_AGGIORNATO' if history_stale else 'OK',
                'ValidazioneSisal':'ASSENTE',
                'FonteProbabilita':'Pool_Bet365_NON_VALIDATO_SISAL' if profile else 'MercatoSisalNoVig',
            })
    if not rows:
        raise ValueError('Non ci sono partite future riconosciute da analizzare. '
                         'Controlla data del PDF, nomi delle squadre e palinsesto.')
    # Archive the original observations BEFORE adding presentation-only
    # ranking fields, to preserve the existing audit schema.
    from sisal_history import append_forecasts
    report_before_ranking=pd.DataFrame(rows)
    data_dir.mkdir(exist_ok=True,parents=True)
    added_snapshots=append_forecasts(report_before_ranking,data_dir,now)
    report=rank_full_board(report_before_ranking)
    quote_csv=data_dir/'sisal_quote_importate.csv'
    result_csv=data_dir/'sisal_analisi.csv'
    result_md=data_dir/'sisal_report.md'
    pd.DataFrame(fixtures).to_csv(quote_csv,index=False)
    report.to_csv(result_csv,index=False)
    write_readable_report(report,result_md,updated=updated,now=now,
                          source_url=OFFICIAL_PDF_URL,last_result=last_result,
                          freshness_reason=freshness_reason,history_stale=history_stale,
                          unrecognized=unrecognized)

    print(f'\nSISAL: aggiornato {updated:%d/%m/%Y %H:%M}; partite PDF {len(fixtures)}; '
          f'partite iniziate {skipped_past}; quote analizzate {len(report)}')
    print(terminal_board(report))
    if unrecognized:
        print('ATTENZIONE: squadre sconosciute per il modello DC, '
              'quote Sisal comunque riportate: ' + '; '.join(sorted(set(unrecognized))))
    print(f'\nNuovi snapshot registrati prima del calcio d’inizio: {added_snapshots}')
    print(f'Quote: {quote_csv}\nGraduatoria completa: {result_csv}\nReport leggibile: {result_md}')
    print('Le probabilità di Sisal non dimostrano valore economico; '
          'nessuna puntata consigliata.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('fetch',help='Download historical Serie A CSV seasons')
    s.add_argument('--seasons',nargs='+',default=SEASONS)
    s=sub.add_parser('backtest',help='Walk-forward calibration and locked holdout')
    s.add_argument('--start',default='2025-08-01')
    s.add_argument('--bankroll',type=float,default=60.0)
    s=sub.add_parser('xg-fetch',help='Download sourced xG CSVs, strictly validated on load')
    s.add_argument('--seasons',nargs='+',type=int,default=[2023,2024,2025,2026])
    s=sub.add_parser('xg-backtest',help='Compare xG/Elo hypotheses and bookmaker on same games')
    s.add_argument('--start',default='2025-08-01')
    s=sub.add_parser('sisal-audit',help='Audit logged Sisal odds vs final scores')
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
    if args.command=='xg-fetch':
        from xg_elo_model import fetch_xg, load_xg
        fetch_xg(args.data_dir,args.seasons)
        official=read_history(args.data_dir)
        _,meta=load_xg(args.data_dir,official)
        print('MATCHES VALIDATED:',meta['matched'])
        return
    if args.command=='xg-backtest':
        from xg_elo_model import load_xg, evaluate_xg_elo
        import json
        official=read_history(args.data_dir)
        xg,info=load_xg(args.data_dir,official)
        print('SOURCE COVERAGE:', info['matched'])
        summary=evaluate_xg_elo(official,xg,args.data_dir,start=args.start)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return
    if args.bankroll<=0: raise ValueError('Bankroll must be > 0')
    if args.command=='sisal':
        sisal_analysis(args.pdf,args.data_dir,args.bankroll,no_refresh=args.no_refresh)
        return
    df=read_history(args.data_dir)
    if args.command=='backtest':
        backtest(df,pd.Timestamp(args.start),args.bankroll,args.data_dir)
    elif args.command=='sisal-audit':
        import json
        from sisal_history import audit_history
        print(json.dumps(audit_history(df,args.data_dir),ensure_ascii=False,indent=2))
    else:
        as_of=pd.Timestamp(args.as_of).normalize() if args.as_of else pd.Timestamp.today().normalize()
        predict(df,args.fixtures,as_of,args.bankroll)

if __name__=='__main__':
    main()
