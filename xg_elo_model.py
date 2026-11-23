"""Independent Serie A xG/Elo experimental forecasts.

The pre-declared hyperparameters in this file are NOT selected on the test set.
Understat 2023-2026 and Sportmonks 2026/27 use DIFFERENT xG definitions;
xG form estimates are calculated WITHIN a season/provider, never pooled across.
Historical fixture xG is post-match information. For a fixture on day D all
features use matches with Date strictly earlier than D.
"""
from __future__ import annotations

import io
import math
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE_URL = 'https://www.xgtables.io'
SOURCE_URLS = {
    2023: f'{BASE_URL}/history/serie-a/2023/fixtures.csv',
    2024: f'{BASE_URL}/history/serie-a/2024/fixtures.csv',
    2025: f'{BASE_URL}/history/serie-a/2025/fixtures.csv',
    2026: f'{BASE_URL}/data/v1/serie-a/2026/fixtures.csv',
}
PROVIDERS = {2023: 'Understat', 2024:'Understat', 2025:'Understat',
             2026: 'Sportmonks'}
ALIAS = {
    'AC Milan':'Milan', 'Parma Calcio 1913':'Parma',
    'Internazionale':'Inter', 'Inter Milan':'Inter',
    'AS Roma':'Roma', 'Hellas Verona':'Verona',
    'SSC Napoli':'Napoli', 'Como 1907':'Como',
}
ELO_K = 20.0
ELO_HOME_BONUS = 65.0
ELO_GOAL_SCALE = 700.0
RECENT_GAMES = 12
HALF_LIFE_DAYS = 90.0
SHRINK_GAMES = 6.0
MIN_XG_TEAM_GAMES = 4
MIN_XG_LEAGUE_GAMES = 45
MAX_BYTES = 2_500_000
GOAL_CAP = 12
SCORE_PATTERN = re.compile(r'^\s*(\d{1,2})\s*[-–—:]\s*(\d{1,2})\s*$')


class XGDataError(ValueError):
    """No silent substitution or fake missing xG observations."""


def fetch_xg(destination: Path, seasons=(2023,2024,2025,2026)) -> list[Path]:
    destination.mkdir(parents=True,exist_ok=True)
    paths=[]
    for season in seasons:
        if season not in SOURCE_URLS:
            raise XGDataError(f'Unsupported season: {season}')
        req=urllib.request.Request(SOURCE_URLS[season],
                                   headers={'User-Agent':'Mozilla/5.0 (academic analysis)',
                                            'Accept':'text/csv'})
        try:
            with urllib.request.urlopen(req,timeout=30) as stream:
                payload=stream.read(MAX_BYTES+1)
        except Exception as exc:
            raise XGDataError(f'xG unavailable for season {season}: {exc}') from exc
        if len(payload)>MAX_BYTES or not payload:
            raise XGDataError(f'Invalid xG CSV for {season}')
        try:
            preview=pd.read_csv(io.BytesIO(payload),nrows=2)
        except Exception as exc:
            raise XGDataError(f'Invalid CSV encoding/schema {season}: {exc}') from exc
        if not {'date','home','away','home_xg','away_xg'}.issubset(preview.columns):
            raise XGDataError(f'xG CSV schema changed {season}: {preview.columns.tolist()}')
        name=destination/f'xg_seriea_{season}_{PROVIDERS[season].lower()}.csv'
        name.write_bytes(payload)
        paths.append(name)
        print(f'XG verified download: {season} / {PROVIDERS[season]} / {len(payload)} bytes')
    return paths


def _score(record: pd.Series) -> tuple[int,int] | None:
    score=str(record.get('score','')).strip()
    m=SCORE_PATTERN.fullmatch(score)
    if m:
        return int(m.group(1)),int(m.group(2))
    # Current Sportmonks exports can contain a separate 90-minute score.
    for a,b in [('home_goals_90','away_goals_90'),
                ('home_score_90','away_score_90'),
                ('home_score','away_score')]:
        if a in record.index and b in record.index:
            va=pd.to_numeric(record[a],errors='coerce')
            vb=pd.to_numeric(record[b],errors='coerce')
            if pd.notna(va) and pd.notna(vb) and float(va).is_integer() and float(vb).is_integer():
                return int(va),int(vb)
    return None


def load_xg(directory: Path, official: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    """Strict reconciliation with Football-Data results: no conflicting scores."""
    paths=sorted(directory.glob('xg_seriea_*.csv'))
    if not paths:
        raise XGDataError('No local xG CSVs. Run xg-fetch before evaluation.')
    keys=['Date','HomeTeam','AwayTeam']
    ref=official[keys+['FTHG','FTAG']].copy()
    ref['Date']=pd.to_datetime(ref.Date).dt.normalize()
    if ref.duplicated(keys).any():
        raise XGDataError('Duplicate official fixtures, cannot reconcile xG')
    normalized=[]
    stats={'downloaded_seasons':[],'matched':{},'ignored_unmatched':{},
           'ignored_missing_xg':{},'source_urls':SOURCE_URLS}
    for path in paths:
        m=re.fullmatch(r'xg_seriea_(\d{4})_(understat|sportmonks)\.csv',path.name)
        if not m:
            continue
        season=int(m.group(1))
        provider=m.group(2).title()
        if season not in PROVIDERS or PROVIDERS[season] != provider:
            raise XGDataError(f'Unexpected source name for xG file {path.name}')
        data=pd.read_csv(path,low_memory=False)
        required={'date','home','away','home_xg','away_xg'}
        if data.empty or not required.issubset(data.columns):
            raise XGDataError(f'Malformed xG fixture file {path.name}')
        data['Date']=pd.to_datetime(data.date,errors='coerce',format='mixed').dt.normalize()
        if data.Date.isna().any():
            raise XGDataError(f'Unparseable fixture dates in {path.name}')
        # The 2026 source has a different xG model: this provider tag
        # prevents cross-season aggregation of incompatible xG metrics.
        data['HomeTeam']=data.home.astype(str).str.strip().replace(ALIAS)
        data['AwayTeam']=data.away.astype(str).str.strip().replace(ALIAS)
        data['XGH']=pd.to_numeric(data.home_xg,errors='coerce')
        data['XGA']=pd.to_numeric(data.away_xg,errors='coerce')
        unreliable=data.get('totals_unreliable',pd.Series(False,index=data.index))
        bad_flag=unreliable.astype(str).str.strip().str.lower().isin(['true','1','yes'])
        valid=(data.XGH.notna()&data.XGA.notna() &
               data.XGH.between(0,12)&data.XGA.between(0,12)&~bad_flag)
        scored=data.apply(_score,axis=1)
        data['ScoreH']=[v[0] if v is not None else np.nan for v in scored]
        data['ScoreA']=[v[1] if v is not None else np.nan for v in scored]
        valid &= data.ScoreH.notna() & data.ScoreA.notna()
        clean=data.loc[valid,keys+['XGH','XGA','ScoreH','ScoreA']].copy()
        if clean.duplicated(keys).any():
            raise XGDataError(f'Duplicate xG matches in {path.name}')
        clean=clean.merge(ref,on=keys,how='inner',validate='one_to_one')
        if ((clean.ScoreH!=clean.FTHG)|(clean.ScoreA!=clean.FTAG)).any():
            bad=clean[(clean.ScoreH!=clean.FTHG)|(clean.ScoreA!=clean.FTAG)].iloc[0]
            raise XGDataError(f'xG/official score conflict: {bad.HomeTeam} - {bad.AwayTeam} {bad.Date}')
        clean['Provider']=PROVIDERS[season]
        clean['Season']=season
        normalized.append(clean[keys+['XGH','XGA','Provider','Season']])
        stats['downloaded_seasons'].append(season)
        stats['matched'][str(season)]=len(clean)
        stats['ignored_unmatched'][str(season)]=max(0,int(valid.sum())-len(clean))
        stats['ignored_missing_xg'][str(season)]=int((~valid).sum())
    if not normalized:
        raise XGDataError('No xG matches with verified official results.')
    matched=pd.concat(normalized,ignore_index=True).sort_values(keys).reset_index(drop=True)
    if matched.duplicated(keys).any():
        raise XGDataError('Same fixture is present in multiple xG seasons.')
    return matched,stats


def update_elo_day(ratings: dict[str,float], day: pd.DataFrame) -> None:
    """Apply all same-date scores simultaneously after their forecasts."""
    changes:dict[str,float]={}
    for row in day.itertuples():
        home,away=str(row.HomeTeam),str(row.AwayTeam)
        rh,ra=ratings.get(home,1500.0),ratings.get(away,1500.0)
        expectation=1.0/(1.0+10**(-(rh+ELO_HOME_BONUS-ra)/400))
        outcome=1.0 if row.FTHG>row.FTAG else (0.5 if row.FTHG==row.FTAG else 0.)
        delta=ELO_K*(outcome-expectation)
        changes[home]=changes.get(home,0.)+delta
        changes[away]=changes.get(away,0.)-delta
    for team,delta in changes.items():
        ratings[team]=ratings.get(team,1500.0)+delta


def elo_ratings(history: pd.DataFrame) -> dict[str,float]:
    """Pre-match Elo updated in date batches; same-day results are symmetric."""
    ratings:dict[str,float]={}
    for _,day in history.sort_values(['Date','HomeTeam','AwayTeam']).groupby('Date',sort=True):
        update_elo_day(ratings, day)
    return ratings


def _scoring_rates(history: pd.DataFrame) -> tuple[float,float]:
    if len(history)<40:
        return 1.45,1.15  # Fixed structural priors; not actual observed data.
    recent=history.sort_values('Date').tail(500)
    return max(.4,float(recent.FTHG.mean())),max(.4,float(recent.FTAG.mean()))


def _score_probs(home_goals: float,away_goals: float) -> dict[str,float]:
    if not (np.isfinite(home_goals) and np.isfinite(away_goals) and
            .15<=home_goals<=5 and .15<=away_goals<=5):
        raise ValueError('Invalid expected-goal intensity')
    axis=np.arange(GOAL_CAP+1)
    scores=np.outer(poisson.pmf(axis,home_goals),poisson.pmf(axis,away_goals))
    scores /= scores.sum()
    h,a=np.indices(scores.shape)
    return {'1':float(scores[h>a].sum()),'X':float(scores[h==a].sum()),
            '2':float(scores[h<a].sum()),
            'OVER2.5':float(scores[h+a>=3].sum()),
            'UNDER2.5':float(scores[h+a<=2].sum())}


def forecast_asof(history: pd.DataFrame, xg: pd.DataFrame, cutoff: pd.Timestamp,
                  home: str, away: str, *,
                  _precomputed_ratings: dict[str,float] | None = None) -> dict:
    """Independent hypotheses, strictly pre-date; no bookmaker probabilities."""
    cutoff=pd.Timestamp(cutoff).normalize()
    past=history[history.Date<cutoff]
    if len(past)<300:
        return {'status':'INSUFFICIENT_GOALS_HISTORY'}
    # Evaluation supplies the Elo snapshot built BEFORE the current date;
    # direct calls independently rebuild the same snapshot from past results.
    ratings = (_precomputed_ratings if _precomputed_ratings is not None
               else elo_ratings(past))
    base_h,base_a=_scoring_rates(past)
    diff=ratings.get(home,1500.)-ratings.get(away,1500.)
    modifier=float(np.clip(diff/ELO_GOAL_SCALE,-.6,.6))
    elo_h=float(np.clip(base_h*math.exp(modifier),.2,4.9))
    elo_a=float(np.clip(base_a*math.exp(-modifier),.2,4.9))
    prediction={'status':'ELO_ONLY','elo':_score_probs(elo_h,elo_a),
                'elo_home_goals':elo_h,'elo_away_goals':elo_a,
                'xg':None,'hybrid':None,'provider':None,
                'n_home_xg':0,'n_away_xg':0}
    if xg.empty:
        return prediction
    season=cutoff.year if cutoff.month>=7 else cutoff.year-1
    provider=PROVIDERS.get(season)
    if not provider:
        return prediction
    prior=xg[(xg.Date<cutoff)&(xg.Season==season)&(xg.Provider==provider)]
    if len(prior)<MIN_XG_LEAGUE_GAMES:
        return prediction
    average=float((prior.XGH.sum()+prior.XGA.sum())/(2*len(prior)))
    mean_home=float(prior.XGH.mean())
    mean_away=float(prior.XGA.mean())
    if not (.25<average<4.5 and .25<mean_home<4.5 and .25<mean_away<4.5):
        return prediction

    def club_strength(team: str) -> tuple[float,float,int]:
        club=prior[(prior.HomeTeam==team)|(prior.AwayTeam==team)].sort_values('Date').tail(RECENT_GAMES)
        if club.empty:
            return average,average,0
        attack=np.where(club.HomeTeam==team,club.XGH,club.XGA)
        conceded=np.where(club.HomeTeam==team,club.XGA,club.XGH)
        ages=(cutoff-club.Date).dt.days.to_numpy(dtype=float)
        weight=np.exp(-math.log(2)*ages/HALF_LIFE_DAYS)
        attack_est=(float(np.dot(weight,attack))+SHRINK_GAMES*average)/(weight.sum()+SHRINK_GAMES)
        defense_est=(float(np.dot(weight,conceded))+SHRINK_GAMES*average)/(weight.sum()+SHRINK_GAMES)
        return attack_est,defense_est,len(club)

    att_h,def_h,n_h=club_strength(home)
    att_a,def_a,n_a=club_strength(away)
    prediction.update({'n_home_xg':n_h,'n_away_xg':n_a,'provider':provider})
    if min(n_h,n_a)<MIN_XG_TEAM_GAMES:
        return prediction
    xgh=float(np.clip(mean_home*(att_h/average)*(def_a/average),.2,4.9))
    xga=float(np.clip(mean_away*(att_a/average)*(def_h/average),.2,4.9))
    # Two predeclared fixed-form hypotheses, no tuning on evaluated games.
    # Geometric mean on the goal-intensity scale, not a blended bookmaker price.
    mix_h=math.sqrt(xgh*elo_h)
    mix_a=math.sqrt(xga*elo_a)
    prediction.update({'status':'XG_ELO_AVAILABLE',
                       'xg':_score_probs(xgh,xga),
                       'hybrid':_score_probs(mix_h,mix_a),
                       'xg_home_goals':xgh,'xg_away_goals':xga,
                       'hybrid_home_goals':mix_h,'hybrid_away_goals':mix_a})
    return prediction


def evaluate_xg_elo(history: pd.DataFrame, xg: pd.DataFrame,
                    directory: Path, *, start: str='2025-08-01') -> dict:
    """Compare every comparable forecast with market on the exact same games.

    This is exploratory: 2025/26 has previously been seen in model development,
    and the 2026/27 provider differs. Do not choose a winner by retrospective ROI.
    """
    from seriea_value_model import odds_from_row, no_vig
    start_date=pd.Timestamp(start)
    if start_date<pd.Timestamp('2025-08-01'):
        raise ValueError('The evaluation start is fixed at 2025-08-01 or later')
    evaluated=[]
    # Incremental ratings are mathematically equivalent to repeatedly replaying
    # all prior games, but avoid hundreds of expensive groupby reconstructions.
    ratings=elo_ratings(history[history.Date<start_date])
    for date, matches in history[history.Date>=start_date].groupby('Date',sort=True):
        day=pd.Timestamp(date)
        for _,row in matches.iterrows():
            prices=odds_from_row(row,price_kind='opening')
            market=no_vig(prices,('1','X','2'))
            if not market:
                continue
            fc=forecast_asof(history,xg,day,row.HomeTeam,row.AwayTeam,
                             _precomputed_ratings=ratings)
            target='1' if row.FTHG>row.FTAG else ('X' if row.FTHG==row.FTAG else '2')
            for method, probs in [('Elo',fc.get('elo')),
                                  ('xG',fc.get('xg')),
                                  ('xG+Elo',fc.get('hybrid')),
                                  ('Bet365 apertura',market)]:
                if probs is None:
                    continue
                evaluated.append({'Date':str(day.date()),'Home':row.HomeTeam,
                                  'Away':row.AwayTeam,'Season':day.year if day.month>=7 else day.year-1,
                                  'Provider':fc.get('provider') or 'Elo only',
                                  'Method':method,'Actual':target,
                                  'p1':probs['1'],'pX':probs['X'],'p2':probs['2'],
                                  'log_loss':-math.log(max(probs[target],1e-12)),
                                  'brier':sum((probs[v]-float(v==target))**2
                                              for v in ('1','X','2'))})
        # Only now, after EVERY fixture of this date has been predicted,
        # may these results update the ratings used on the next date.
        update_elo_day(ratings, matches)
    if not evaluated:
        raise XGDataError('No comparable evaluation fixtures')
    detailed=pd.DataFrame(evaluated)
    directory.mkdir(parents=True,exist_ok=True)
    detailed.to_csv(directory/'xg_elo_predictions.csv',index=False)
    summary=detailed.groupby(['Season','Method'],as_index=False).agg(
        n=('log_loss','size'),log_loss=('log_loss','mean'),brier=('brier','mean'))
    summary.to_csv(directory/'xg_elo_metrics.csv',index=False)
    # Comparisons restricted to exact same matches across all four methods.
    pivot=detailed.pivot_table(index=['Date','Home','Away'],columns='Method',
                               values='log_loss',aggfunc='first')
    overlap=pivot.dropna(subset=['Elo','xG','xG+Elo','Bet365 apertura'])
    comparisons={}
    if len(overlap):
        fixture_ids=set(overlap.index)
        subset=detailed.set_index(['Date','Home','Away'])
        matched=subset[subset.index.isin(fixture_ids)]
        for label,frame in matched.groupby('Method'):
            comparisons[label]={'n':int(len(frame)),
                                'log_loss':float(frame.log_loss.mean()),
                                'brier':float(frame.brier.mean())}
    result={'period':'Retrospective 2025/26 previously observed; not untouched holdout',
            'comparisons_same_matches':comparisons,
            'matched_four_way':len(overlap),
            'season_metrics':summary.to_dict('records'),
            'conclusion':'Not validated for Sisal; no betting recommendation'}
    lines=['# Experimental xG + Elo validation','',
           'No post-match xG is used to predict the same match.',
           'xG sources: Understat through 2025/26; Sportmonks in 2026/27.',
           '**These two xG measures are not pooled across seasons.**','',
           'The 2025/26 test period had already been examined in prior experiments,',
           'so these comparisons are retrospective, not an untouched holdout.','',
           '| Season | Model | N | Log loss ↓ | Brier ↓ |',
           '|---|---|---:|---:|---:|']
    for row in result['season_metrics']:
        lines.append(f"| {int(row['Season'])} | {row['Method']} | {int(row['n'])} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    lines+=['','## Equal-match comparison','','| Method | N | Log loss ↓ | Brier ↓ |',
            '|---|---:|---:|---:|']
    for name,score in comparisons.items():
        lines.append(f"| {name} | {score['n']} | {score['log_loss']:.4f} | {score['brier']:.4f} |")
    if not comparisons:
        lines.append('| — | 0 | n/d | n/d |')
    lines+=['','Differences in log loss do not establish a positive betting ROI.',
            'No model has been selected based on retrospective profit.','']
    (directory/'xg_elo_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
