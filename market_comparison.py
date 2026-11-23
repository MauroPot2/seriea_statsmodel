"""Bookmaker consensus diagnostics, never a verified wagering edge.

HISTORICAL: Football-Data opening quotes are not simultaneous snapshots.
LIVE: user-supplied price observations must have real, timezone-aware timestamps
within a 120-second window, before kickoff. No undocumented scraping/API.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = {
    '1X2': ('1', 'X', '2'),
    'OU25': ('OVER2.5', 'UNDER2.5'),
}
# Explicit per-bookmaker columns only. Max/Avg/BFE exchange are NOT bookmakers.
# Pinnacle public feed has known stale-data issues from 2025-07-23; exclude it.
BOOKS_1X2 = {
    'B365': ('B365H','B365D','B365A'),
    'BW': ('BWH','BWD','BWA'),
    'IW': ('IWH','IWD','IWA'),
    'PS': ('PSH','PSD','PSA'),
    'WH': ('WHH','WHD','WHA'),
    'VC': ('VCH','VCD','VCA'),
    'BF': ('BFH','BFD','BFA'),
    'BFD': ('BFDH','BFDD','BFDA'),
    'BMG': ('BMGMH','BMGMD','BMGMA'),
    'BV': ('BVH','BVD','BVA'),
    'CL': ('CLH','CLD','CLA'),
    'LB': ('LBH','LBD','LBA'),
    '1XB': ('1XBH','1XBD','1XBA'),
    'PP': ('PPH','PPD','PPA'),
    'SKB': ('SKBH','SKBD','SKBA'),
}
BOOKS_OU25 = {
    'B365': ('B365>2.5','B365<2.5'),
    'PS': ('P>2.5','P<2.5'),
}
P_DEFAULT = pd.Timestamp('2025-07-23')
MIN_OTHER_BOOKS = 3
MAX_CAPTURE_SKEW_SECONDS = 120
# Descriptive proxies only. No trade or gain is inferred.
DISCLAIMER = (
    'PREZZI INDICATIVI, NON SCOMMESSE CONVENIENTI VALIDATE: '
    'le probabilità sono ricavate da altri bookmaker, non sono esiti veri. '
    'Le quote storiche potrebbero non essere state simultaneamente disponibili.'
)


def eligible_book(book: str, date: pd.Timestamp) -> bool:
    """Known stale Pinnacle public API: do not trust PS/P after 2025-07-22."""
    return not (book == 'PS' and pd.Timestamp(date).normalize() >= P_DEFAULT)


def _price(value: object) -> float | None:
    parsed = pd.to_numeric(value, errors='coerce')
    if pd.isna(parsed) or not np.isfinite(parsed):
        return None
    result = float(parsed)
    return result if 1.0 < result <= 1000.0 else None


def no_vig(prices: dict[str, float]) -> dict[str, float]:
    if not prices or any(_price(v) is None for v in prices.values()):
        raise ValueError('Incomplete or invalid price set')
    inverse = {key: 1.0 / value for key, value in prices.items()}
    total = sum(inverse.values())
    if not 0 < total < 10:
        raise ValueError('Unusable bookmaker probabilities')
    return {key: value / total for key, value in inverse.items()}


def historical_books(row: pd.Series, group: str = '1X2') -> dict[str, dict[str,float]]:
    """Only complete 1X2 or U/O bookmaker odds, no Max/Avg proxies."""
    specs = BOOKS_1X2 if group == '1X2' else BOOKS_OU25
    date = pd.Timestamp(row['Date'])
    result = {}
    for book, columns in specs.items():
        if not eligible_book(book, date):
            continue
        values = [_price(row.get(c)) for c in columns]
        if all(value is not None for value in values):
            result[book] = dict(zip(OUTCOMES[group],values))
    return result


def independent_consensus(prices_by_book: dict[str,dict[str,float]],
                          excluded_book: str | None = None,
                          min_other_books: int = MIN_OTHER_BOOKS) -> dict | None:
    """Median no-vig probability from OTHER books. Never use the target price."""
    available = {name:no_vig(odds) for name,odds in prices_by_book.items()
                 if name != excluded_book}
    if len(available) < min_other_books:
        return None
    keys = tuple(next(iter(available.values())))
    if any(set(probabilities) != set(keys) for probabilities in available.values()):
        raise ValueError('Incompatible market outcomes')
    matrix = np.array([[probabilities[key] for key in keys]
                       for probabilities in available.values()],dtype=float)
    medians = np.median(matrix,axis=0)
    medians /= medians.sum()
    floor = np.min(matrix,axis=0)
    ceiling = np.max(matrix,axis=0)
    return {'probabilities':dict(zip(keys,medians)),
            'min_probabilities':dict(zip(keys,floor)),
            'max_probabilities':dict(zip(keys,ceiling)),
            'n_others':len(available),
            'source_books':sorted(available)}


def comparative_prices(prices_by_book: dict[str,dict[str,float]],
                       *, min_other_books: int = MIN_OTHER_BOOKS) -> list[dict]:
    """Every outcome/book with leave-one-out market estimate, without selection bias.

    A positive price premium is NOT an expected value estimate with validated
    probabilities, nor a prospective betting recommendation.
    """
    if not prices_by_book:
        return []
    results = []
    for book, prices in sorted(prices_by_book.items()):
        comparison = independent_consensus(prices_by_book,book,min_other_books)
        for selection, price in prices.items():
            p = comparison['probabilities'][selection] if comparison else None
            pessimistic = comparison['min_probabilities'][selection] if comparison else None
            results.append({
                'bookmaker':book, 'market':selection, 'price':price,
                'p_market_loo':p,
                'n_others':comparison['n_others'] if comparison else 0,
                'market_price_premium':(p*price-1) if p is not None else None,
                # Not a statistical confidence interval: one observed book's
                # lowest probability, a crude adverse-market scenario ONLY.
                'adverse_market_scenario':(pessimistic*price-1) if pessimistic is not None else None,
                'source_books':','.join(comparison['source_books']) if comparison else '',
                'status':'RETROSPECTIVE_UNSYNCHRONIZED' if comparison else 'INSUFFICIENT_INDEPENDENT_BOOKS',
            })
    return results


def _scores(p: dict[str,float], outcome: str) -> tuple[float,float]:
    if outcome not in p:
        raise ValueError('Unknown actual outcome')
    actual = {name:int(name==outcome) for name in p}
    return -math.log(p[outcome]),sum((p[name]-actual[name])**2 for name in p)


def compare_historical(df: pd.DataFrame, directory: Path,
                       start: str = '2025-08-01') -> dict:
    """Fixed paired benchmark, with every row and error retained.

    This is a RETROSPECTIVE analysis using previously-examined history.
    Football-Data opening book quotes need not share an exact timestamp.
    """
    from model_evaluation import HOLDOUT_START
    cutoff = pd.Timestamp(start).normalize()
    if cutoff < HOLDOUT_START:
        raise ValueError('Historical comparison starts no earlier than 2025-08-01')
    directory.mkdir(parents=True,exist_ok=True)
    premiums = []
    accuracy = []
    missing = 0
    for _, row in df[df.Date >= cutoff].iterrows():
        prices=historical_books(row,'1X2')
        winner=('1' if row.FTHG>row.FTAG else
                ('X' if row.FTHG==row.FTAG else '2'))
        base={'date':str(row.Date.date()),'home':row.HomeTeam,'away':row.AwayTeam,
              'real_outcome':winner}
        if len(prices)<MIN_OTHER_BOOKS+1:
            missing += 1
        for offer in comparative_prices(prices):
            premiums.append({**base,**offer})
        if 'B365' not in prices:
            continue
        consensus=independent_consensus(prices,'B365')
        if consensus is None:
            continue
        p_b365=no_vig(prices['B365'])
        for method,probabilities in (
            ('B365_open_no_vig',p_b365),
            ('Consensus_other_books',consensus['probabilities']),
        ):
            ll,brier=_scores(probabilities,winner)
            accuracy.append({**base,'method':method,'n_other_books':consensus['n_others'],
                             'log_loss':ll,'brier':brier})
    quote_cols=['date','home','away','real_outcome','bookmaker','market','price',
                'p_market_loo','n_others','market_price_premium',
                'adverse_market_scenario','source_books','status']
    diagnostic=pd.DataFrame(premiums,columns=quote_cols)
    diagnostic.to_csv(directory/'market_historical_prices.csv',index=False)
    all_scores=pd.DataFrame(accuracy,
                            columns=['date','home','away','real_outcome','method',
                                     'n_other_books','log_loss','brier'])
    all_scores.to_csv(directory/'market_historical_scores.csv',index=False)
    metrics={}
    for name,part in all_scores.groupby('method',sort=True):
        metrics[name]={'matches':int(len(part)),
                       'log_loss':float(part.log_loss.mean()),
                       'brier':float(part.brier.mean())}
    # Pair by date and fixture for day-cluster bootstrap of score *difference*
    # (not betting ROI). A negative Δ means consensus is more accurate.
    bootstrap = None
    if metrics and all_scores.date.nunique()>=20:
        frame=all_scores.pivot(index=['date','home','away'],
                               columns='method',values='log_loss').dropna()
        frame['delta']=(frame.Consensus_other_books - frame.B365_open_no_vig)
        day_values=frame.groupby(level='date').delta.agg(['sum','count']).to_numpy()
        seed=np.random.default_rng(73421)
        drawn=seed.integers(len(day_values),size=(2000,len(day_values)))
        samples=day_values[drawn].sum(axis=1)
        ranges=samples[:,0]/samples[:,1]
        bootstrap=[float(x) for x in np.quantile(ranges,[.025,.975])]
    result={'price_source':'Football-Data historical opening estimates',
            'not_synchronized':True,
            'pinnacle_stale_from':'2025-07-23',
            'n_missing_four_books':missing,
            'n_matches_compared':int(len(all_scores)//2),
            'accuracy':metrics,
            'day_bootstrap_95_delta_log_loss':bootstrap,
            'positive_return_validated':False,
            'recommendation':'NO_REAL_BETTING_INFERENCE'}
    report=[
        '# Diagnostic of multi-bookmaker odds',
        '',
        'Historical Football-Data opening quotes, **NOT synchronized snapshots**.',
        'Pinnacle/PS excluded from 23 July 2025 because its public odds feed has '
        'reported delays.',
        '',
        '**The market consensus is not independent ground truth.** The premium '
        'is a disagreement between prices, not a verified economic advantage.',
        'Data already examined in previous modelling: this is exploratory and '
        'not an untouched test.',
        '',
        '| Benchmark | Fixtures | Log loss ↓ | Brier ↓ |',
        '|---|---:|---:|---:|',
    ]
    for name,stats in metrics.items():
        report.append(f"| {name} | {stats['matches']} | {stats['log_loss']:.4f} | {stats['brier']:.4f} |")
    if bootstrap is not None:
        report.append(f'\nDay-cluster bootstrap 95% interval, consensus - B365 log loss: '
                      f'[{bootstrap[0]:+.4f}, {bootstrap[1]:+.4f}].')
    report.extend(['',
        f'Comparable fixtures: {result["n_matches_compared"]}.',
        f'Fixtures without four bookmaker snapshots: {missing}.',
        '',
        'All investigated prices, **including negative discrepancies**, are in '
        'market_historical_prices.csv. Full paired per-fixture loss figures '
        'are in market_historical_scores.csv.',
        '',
        'A positive price premium is **not** evidence of profitability. '
        'No proof these bookmaker opening prices were available at one time.',
        '',
    ])
    (directory/'market_historical_report.md').write_text('\n'.join(report),encoding='utf-8')
    return result


def read_live_snapshot(path: Path) -> pd.DataFrame:
    """Require independently observed, complete, timezone-aware price snapshots.

    Required columns:
      captured_at,kickoff,home,away,bookmaker,market_group,outcome,odds
    Each book has 3 simultaneous 1X2, or 2 simultaneous U/O selections.
    Within a fixture+market, all bookmakers must be captured within 120 s.
    """
    frame=pd.read_csv(path,dtype={'bookmaker':'string','market_group':'string',
                                  'outcome':'string','home':'string','away':'string'})
    required={'captured_at','kickoff','home','away','bookmaker','market_group',
              'outcome','odds'}
    if frame.empty or not required.issubset(frame):
        raise ValueError('Missing required live snapshot columns or no rows')
    for field in ('home','away','bookmaker','market_group','outcome'):
        if frame[field].isna().any() or frame[field].str.strip().eq('').any():
            raise ValueError(f'Missing value for {field}')
        frame[field]=frame[field].str.strip()
    parsed={}
    for col in ('captured_at','kickoff'):
        try:
            if frame[col].isna().any():
                raise ValueError('missing timestamp')
            ts=pd.to_datetime(frame[col],utc=True,errors='raise')
            # Reject naive or mixed explicit timezone-less timestamps.
            if not frame[col].astype(str).str.contains(r'(Z|[+-]\d\d:\d\d)$',regex=True).all():
                raise ValueError('Timezone offset required')
            parsed[col]=ts
        except (TypeError,ValueError) as exc:
            raise ValueError(f'Invalid timestamp {col} (ISO-8601 with timezone required)') from exc
    frame['captured_at']=parsed['captured_at']
    frame['kickoff']=parsed['kickoff']
    if (frame.captured_at>=frame.kickoff).any():
        raise ValueError('Post-kickoff observations cannot be included')
    if (frame.captured_at>pd.Timestamp.now(tz='UTC')+pd.Timedelta(minutes=2)).any():
        raise ValueError('Future-dated market observations cannot be trusted')
    if (frame.home==frame.away).any():
        raise ValueError('Same home and away team')
    if frame.market_group.map(OUTCOMES).isna().any():
        raise ValueError('Unsupported market group')
    for group,allowed in OUTCOMES.items():
        if (~frame.loc[frame.market_group==group,'outcome'].isin(allowed)).any():
            raise ValueError('Wrong outcome for market group')
    frame['odds']=pd.to_numeric(frame['odds'],errors='coerce')
    if any(_price(value) is None for value in frame.odds):
        raise ValueError('Invalid or missing quote')
    identity=['kickoff','home','away','market_group','bookmaker','outcome']
    if frame.duplicated(identity).any():
        raise ValueError('Duplicate quote for the same snapshot/book/market')
    return frame


def compare_live_snapshot(frame: pd.DataFrame, directory: Path) -> tuple[pd.DataFrame, dict]:
    """Full book comparison, with market-consensus *proxy* and strict clock check."""
    directory.mkdir(parents=True,exist_ok=True)
    output=[]
    rejects=[]
    for key, group in frame.groupby(['kickoff','home','away','market_group'],sort=True):
        kickoff,home,away,market_group=key
        time_gap=(group.captured_at.max()-group.captured_at.min()).total_seconds()
        if time_gap>MAX_CAPTURE_SKEW_SECONDS:
            rejects.append(f'{home} - {away} / {market_group}: '
                           f'capture gap {time_gap:.0f} sec > 120')
            continue
        book_odds={}
        for book, records in group.groupby('bookmaker'):
            if set(records.outcome)!=set(OUTCOMES[market_group]):
                rejects.append(f'{home} - {away} / {market_group} / {book}: incomplete')
                continue
            book_odds[str(book)]=dict(zip(records.outcome,records.odds))
        if not book_odds:
            continue
        for offer in comparative_prices(book_odds):
            # Live prices must pass both the 120s match window and pre-kickoff
            # check, but are still self-reported rather than independently
            # verified by an authorized odds feed.
            output.append({
                'kickoff':kickoff.isoformat(),'home':home,'away':away,
                'market_group':market_group,
                'captured_earliest':group.captured_at.min().isoformat(),
                'captured_latest':group.captured_at.max().isoformat(),
                **offer,
                'status':('USER_RECORDED_UNVERIFIED' if offer['p_market_loo'] is not None
                          else 'INSUFFICIENT_INDEPENDENT_BOOKS'),
            })
    columns=['kickoff','home','away','market_group','captured_earliest',
             'captured_latest','bookmaker','market','price','p_market_loo',
             'n_others','market_price_premium','adverse_market_scenario',
             'source_books','status']
    board=pd.DataFrame(output,columns=columns)
    # Sorting ranks quote disagreement, NOT probability of profitable bets.
    board=board.sort_values(['market_price_premium','kickoff','home','market'],
                            ascending=[False,True,True,True],
                            na_position='last').reset_index(drop=True)
    board.to_csv(directory/'market_live_comparison.csv',index=False)
    lines=[
        '# Comparazione quote raccolte da te',
        '',
        f'Completi: {len(board)} prezzi; gruppi esclusi: {len(rejects)}.',
        '',
        'I timestamp e i prezzi sono **dichiarati dall’utente**: il programma '
        'non li verifica presso i siti dei bookmaker.',
        'Differenza = P consenso altri bookmaker × quota - 1. '
        '**Non è un EV statisticamente validato**.',
        'Scenario avverso = probabilità minima osservata negli altri '
        'bookmaker × quota - 1: non è un intervallo di confidenza.',
        '',
        '| Partita | Mercato | Fonte | Quota | P altri book | Gap indicativo | Altri book |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for _,r in board.iterrows():
        p=f'{r.p_market_loo:.1%}' if pd.notna(r.p_market_loo) else 'n/d'
        gap=f'{r.market_price_premium:+.1%}' if pd.notna(r.market_price_premium) else 'n/d'
        lines.append(f'| {r.home} - {r.away} | {r.market} | {r.bookmaker} | '
                     f'{r.price:.2f} | {p} | {gap} | {int(r.n_others)} |')
    if rejects:
        lines.extend(['','## Esclusioni temporali / quote incomplete',''])
        lines.extend(f'- {issue}' for issue in rejects)
    lines.extend(['','**Nessuna puntata validata, nessuna promessa di rendimento.**',''])
    (directory/'market_live_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return board,{'prices':len(board),'excluded_groups':rejects,
                  'independent_market_probability_validated':False}
