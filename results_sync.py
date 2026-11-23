"""Validated supplementary Serie A match results (research only).

Football-Data is the primary dataset. open-matchday is a secondary daily CSV,
used only to fill missing completed fixtures without overriding existing scores.
Fail closed on conflicts, unknown teams, malformed files, or overdue fixtures.
"""
from __future__ import annotations

import io
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

SOURCE_URL = 'https://github.com/oladeeayo/open-matchday'
BASE_URL = ('https://raw.githubusercontent.com/oladeeayo/open-matchday/'
            'master/data/ITA_1_serie-a')
RESULTS_URL = f'{BASE_URL}/results_2627.csv'
FIXTURES_URL = f'{BASE_URL}/fixtures_2627.csv'
MAX_CSV_BYTES = 2_000_000
ALIASES = {'Internazionale': 'Inter', 'AC Milan': 'Milan',
           'Hellas Verona': 'Verona', 'AS Roma': 'Roma',
           'SSC Napoli': 'Napoli'}


class ResultSyncError(ValueError):
    """Secondary results cannot be trusted; do not propose wagers."""


def download_csv(url: str, required: set[str]) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (educational match analysis)'})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            payload = response.read(MAX_CSV_BYTES + 1)
        if len(payload) > MAX_CSV_BYTES:
            raise ValueError('CSV troppo grande')
        frame = pd.read_csv(io.BytesIO(payload), encoding='utf-8-sig', dtype=str)
    except Exception as exc:
        raise ResultSyncError(f'Impossibile verificare il feed risultati: {url}: {exc}') from exc
    if frame.empty or not required.issubset(frame.columns):
        raise ResultSyncError(f'Formato feed inatteso: {url}')
    return frame


def sync_results(history: pd.DataFrame, now: datetime,
                 data_dir: Path, *,
                 results: pd.DataFrame | None = None,
                 fixtures: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """Add missing matches only after validating a complete secondary schedule.

    Use injected frames for tests. No CSV is written unless checks succeed.
    The source is refreshed daily, not live: today's unreported kickoffs are
    regarded as missing (including live fixtures) and stop recommendations.
    """
    results = results if results is not None else download_csv(
        RESULTS_URL, {'gw', 'date', 'home', 'away', 'ft_home', 'ft_away'})
    fixtures = fixtures if fixtures is not None else download_csv(
        FIXTURES_URL, {'gw', 'date', 'time', 'home', 'away'})
    results = results.copy()
    fixtures = fixtures.copy()

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        frame['Date'] = pd.to_datetime(frame['date'], format='%Y-%m-%d', errors='coerce').dt.normalize()
        if frame['Date'].isna().any():
            raise ResultSyncError('Date non valide nel calendario secondario')
        known = set(history.HomeTeam).union(history.AwayTeam)
        for src, dest in (('home', 'HomeTeam'), ('away', 'AwayTeam')):
            frame[dest] = frame[src].str.strip().replace(ALIASES)
            unknown = set(frame[dest].dropna()) - known
            if unknown or frame[dest].isna().any():
                raise ResultSyncError(f'Squadre non riconosciute nel feed: {sorted(unknown)}')
        frame['gw'] = pd.to_numeric(frame['gw'], errors='coerce')
        if frame['gw'].isna().any():
            raise ResultSyncError('Giornata non valida nel feed')
        return frame

    results = normalize(results)
    fixtures = normalize(fixtures)
    for src, dest in (('ft_home', 'FTHG'), ('ft_away', 'FTAG')):
        results[dest] = pd.to_numeric(results[src], errors='coerce')
        if (results[dest].isna().any() or
            ((results[dest] < 0) | (results[dest] > 20) |
             (results[dest] % 1 != 0)).any()):
            raise ResultSyncError('Risultati mancanti o non plausibili nel feed')
        results[dest] = results[dest].astype(int)
    start = pd.Timestamp('2026-08-01')
    today = pd.Timestamp(now.date())
    results = results[(results.Date >= start) & (results.Date <= today)]
    fixtures = fixtures[fixtures.Date >= start]
    if len(results) < 30:
        raise ResultSyncError('Numero troppo basso di risultati nel feed secondario')

    key_cols = ['Date', 'HomeTeam', 'AwayTeam']
    if results.duplicated(key_cols).any() or fixtures.duplicated(key_cols).any():
        raise ResultSyncError('Partite duplicate nel feed')
    if pd.concat([results[key_cols], fixtures[key_cols]]).duplicated(key_cols).any():
        raise ResultSyncError('Stessa partita presente tra risultati e partite future')
    # Every Serie A gameweek contains 10 fixtures. A missing result must not
    # silently disappear from both completed and future feeds.
    totals = pd.concat([results[['gw']], fixtures[['gw']]]).groupby('gw').size()
    if (totals != 10).any():
        raise ResultSyncError(f'Giornate con numero incontri diverso da 10: {totals[totals != 10].to_dict()}')

    for _, event in fixtures.iterrows():
        kickoff_time = str(event['time']).strip()
        if not pd.notna(event['time']) or len(kickoff_time) != 5:
            # Uncertain time: don't assert completeness for today's fixture.
            if event.Date <= today:
                raise ResultSyncError('Orario di una partita odierna non verificabile')
            continue
        try:
            kickoff = datetime.fromisoformat(f'{event.Date.date()}T{kickoff_time}')
        except ValueError as exc:
            raise ResultSyncError('Orario non valido nel calendario') from exc
        if kickoff <= now:
            raise ResultSyncError('Il calendario secondario contiene una partita già iniziata '
                                  f'priva di risultato: {event.HomeTeam} - {event.AwayTeam}')

    primary = history[history.Date >= start].copy()
    primary_keys = {tuple(row) for row in primary[key_cols].itertuples(index=False, name=None)}
    seen = set()
    new_rows = []
    for _, event in results.iterrows():
        key = (event.Date, event.HomeTeam, event.AwayTeam)
        if key in primary_keys:
            orig = primary[(primary.Date == event.Date) &
                           (primary.HomeTeam == event.HomeTeam) &
                           (primary.AwayTeam == event.AwayTeam)].iloc[0]
            if (int(orig.FTHG), int(orig.FTAG)) != (event.FTHG, event.FTAG):
                raise ResultSyncError(f'Conflitto risultati tra fonti: {event.HomeTeam} - {event.AwayTeam}')
            seen.add(key)
        else:
            new_rows.append({'Date': event.Date, 'HomeTeam': event.HomeTeam,
                             'AwayTeam': event.AwayTeam,
                             'FTHG': event.FTHG, 'FTAG': event.FTAG,
                             'ResultSource': 'open-matchday'})
    if len(seen) < 10:
        raise ResultSyncError('Le due fonti non hanno abbastanza risultati coincidenti')
    added = pd.DataFrame(new_rows, columns=[*key_cols, 'FTHG', 'FTAG', 'ResultSource'])
    enriched = pd.concat([history, added], ignore_index=True)
    enriched = enriched.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    data_dir.mkdir(exist_ok=True, parents=True)
    added.to_csv(data_dir / 'risultati_integrati_open_matchday.csv', index=False)
    last = enriched[enriched.Date < today].Date.max()
    details = (f'Calendario secondario verificato: {len(results)} risultati, '
               f'{len(seen)} coincidenti, {len(added)} integrati; '
               f'ultimo risultato utilizzabile {last.date()}. Fonte: {SOURCE_URL}')
    return enriched, details
