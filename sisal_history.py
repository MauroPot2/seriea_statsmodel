"""Immutable-enough local log of observed, pre-match Sisal forecast snapshots.

This is a research audit: the bookmaker quote and model probability are stored
before kickoff. No post-event probabilities are recomputed for audit rows.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE = 'sisal_forecast_archive.csv'
KEYS = ['Data','Ora','HomeTeam','AwayTeam','Mercato','AggiornamentoSisal']


def append_forecasts(report: pd.DataFrame, directory: Path, now: datetime) -> int:
    """Append once per fixed Sisal PDF timestamp; never overwrite snapshots."""
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/ARCHIVE
    candidate=report.copy()
    candidate['RegistratoIl']=now.isoformat(timespec='seconds')
    kickoffs=pd.to_datetime(candidate.Data+'T'+candidate.Ora,errors='raise')
    candidate=candidate.loc[kickoffs>pd.Timestamp(now)]
    if candidate.empty:
        return 0
    if path.exists():
        existing=pd.read_csv(path,dtype=str)
        if not set(KEYS).issubset(existing.columns):
            raise ValueError('Archivio Sisal esistente con colonne incompatibili')
        # Schema evolution: older snapshots retain their original values.
        # Newly introduced independent forecasts remain missing on old rows,
        # never retroactively recomputed from results known later.
        for column in candidate.columns:
            if column not in existing.columns:
                existing[column] = pd.NA
        keys=set(tuple(rec) for rec in existing[KEYS].itertuples(index=False,name=None))
        candidate=candidate.loc[[tuple(rec) not in keys for rec in
                                 candidate[KEYS].itertuples(index=False,name=None)]]
        if candidate.empty:
            return 0
        full=pd.concat([existing,candidate],ignore_index=True)
    else:
        full=candidate
    temporary=path.with_suffix('.tmp')
    full.to_csv(temporary,index=False)
    temporary.replace(path)
    return len(candidate)


def audit_history(results: pd.DataFrame, directory: Path) -> dict:
    """Score only recorded, complete pre-match market snapshots against results.

    The procedure picks the FIRST observed quote for each fixture/market,
    before knowing whether the later quote became more attractive.
    """
    path=directory/ARCHIVE
    if not path.exists():
        return {'n_matches':0,'message':'Nessuno snapshot Sisal registrato'}
    archive=pd.read_csv(path)
    required=set(KEYS+['RegistratoIl','QuotaSisal','ProbModello','ProbMercatoNoVig',
                       'QualitaStorico','ValidazioneSisal'])
    if not required.issubset(archive.columns):
        raise ValueError('Archivio Sisal incompleto, audit non affidabile')
    archive['Date']=pd.to_datetime(archive.Data,errors='raise').dt.normalize()
    archive['Recorded']=pd.to_datetime(archive.RegistratoIl, errors='raise')
    archive['Kickoff']=pd.to_datetime(archive.Data+'T'+archive.Ora, errors='raise')
    archive=archive[(archive.Recorded < archive.Kickoff) &
                    (archive.QualitaStorico == 'OK')].copy()
    archive=archive.sort_values('Recorded').drop_duplicates(
        ['Date','HomeTeam','AwayTeam','Mercato'],keep='first')
    complete=results[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].drop_duplicates(
        ['Date','HomeTeam','AwayTeam'],keep='last')
    joined=archive.merge(complete,on=['Date','HomeTeam','AwayTeam'],how='inner',
                         validate='many_to_one')
    if joined.empty:
        return {'n_matches':0,'message':'Nessuna partita dello storico Sisal ancora conclusa'}
    joined['Realizzato']=joined.apply(lambda r: int({
        '1': r.FTHG > r.FTAG, 'X':r.FTHG == r.FTAG, '2':r.FTHG < r.FTAG,
        'OVER2.5':r.FTHG+r.FTAG>=3,
        'UNDER2.5':r.FTHG+r.FTAG<=2}[r.Mercato]),axis=1)
    rows=[]
    for group, labels in [('1X2',['1','X','2']),('OU25',['OVER2.5','UNDER2.5'])]:
        subset=joined[joined.Mercato.isin(labels)]
        for _, frame in subset.groupby(['Date','HomeTeam','AwayTeam'],sort=True):
            if len(frame)!=len(labels) or set(frame.Mercato)!=set(labels):
                continue
            frame=frame.set_index('Mercato').loc[labels]
            outcome=frame.Realizzato.to_numpy(dtype=float)
            p=frame.ProbModello.to_numpy(dtype=float)
            m=frame.ProbMercatoNoVig.to_numpy(dtype=float)
            if (not np.all(np.isfinite(p)) or not np.all(np.isfinite(m)) or
                (p<=0).any() or (m<=0).any() or
                abs(p.sum()-1)>0.005 or abs(m.sum()-1)>0.005):
                continue
            rows.append({'group':group,'date':str(frame.Date.iloc[0].date()),
                         'log_loss_model':-float(np.log(p[outcome==1][0])),
                         'log_loss_market':-float(np.log(m[outcome==1][0])),
                         'brier_model':float(np.square(p-outcome).sum()),
                         'brier_market':float(np.square(m-outcome).sum())})
    measurements=pd.DataFrame(rows)
    if measurements.empty:
        return {'n_matches':0,'message':'Nessun mercato completo verificabile'}
    directory.mkdir(parents=True,exist_ok=True)
    measurements.to_csv(directory/'sisal_audit_matches.csv',index=False)
    groups={}
    for group, frame in measurements.groupby('group'):
        groups[group]={'n':int(len(frame)),
                        'model_ll':float(frame.log_loss_model.mean()),
                        'market_ll':float(frame.log_loss_market.mean()),
                        'model_brier':float(frame.brier_model.mean()),
                        'market_brier':float(frame.brier_market.mean())}
    return {'n_matches':int(measurements[['date','group']].drop_duplicates().date.nunique()),
            'groups':groups,
            'status':'SAMPLE_INSUFFICIENT' if min(v['n'] for v in groups.values())<300 else 'RESEARCH_ONLY',
            'message':'Non dimostra un vantaggio economico e non abilita puntate'}
