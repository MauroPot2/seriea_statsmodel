"""Full Sisal odds board, not a wagering recommendation.

Show all parsed outcomes. Primary ranking uses the bookmaker's vig-removed
implied probabilities: historical validation did not establish that the
Poisson forecast was more accurate than the market.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MARKET_GROUP = {
    '1': '1X2', 'X': '1X2', '2': '1X2',
    'OVER2.5': 'Under/Over 2.5', 'UNDER2.5': 'Under/Over 2.5',
}
MARKET_LABEL = {
    '1': 'Vittoria casa', 'X': 'Pareggio', '2': 'Vittoria ospite',
    'OVER2.5': 'Over 2,5', 'UNDER2.5': 'Under 2,5',
}
GROUP_ORDER = {'1X2': 0, 'Under/Over 2.5': 1}


def rank_full_board(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank outcomes within 1X2 and Under/Over independently."""
    if frame.empty:
        raise ValueError('Nessuna quota Sisal disponibile per il report')
    result = frame.copy()
    if result['Mercato'].map(MARKET_GROUP).isna().any():
        raise ValueError('Mercato Sisal non supportato nel report')
    result['GruppoMercato'] = result.Mercato.map(MARKET_GROUP)
    result['DescrizioneEsito'] = result.Mercato.map(MARKET_LABEL)
    if not np.isfinite(result.ProbMercatoNoVig.to_numpy(dtype=float)).all():
        raise ValueError('Probabilità bookmaker mancanti')
    result['RankMercato'] = result.groupby(
        ['Data', 'Ora', 'Partita', 'GruppoMercato'], sort=False
    ).ProbMercatoNoVig.rank(method='first', ascending=False).astype(int)
    result['PiuProbabileNelMercato'] = result.RankMercato == 1
    result['OrdineGruppo'] = result.GruppoMercato.map(GROUP_ORDER)
    return (result.sort_values(
        ['Data', 'Ora', 'Partita', 'OrdineGruppo', 'RankMercato'], kind='stable'
    ).drop(columns=['OrdineGruppo']).reset_index(drop=True))


def _prob(p: float) -> str:
    return f'{float(p):.1%}'


def _raw_probability(value: object) -> str:
    return _prob(value) if pd.notna(value) else 'n/d'


def terminal_board(report: pd.DataFrame) -> str:
    """Compact fixture-by-fixture view containing ALL imported prices."""
    lines = [
        'QUOTE SISAL — SERIE A (tutti gli esiti disponibili)',
        'P = probabilità implicita Sisal SENZA MARGINE (stima del mercato).',
        'DC = previsione Poisson/Dixon–Coles: NON superiore al mercato nel backtest.',
        'Graduatoria solo all’interno di 1X2 o Under/Over; NON suggerisce puntate.',
    ]
    for (_, _, fixture), games in report.groupby(['Data', 'Ora', 'Partita'], sort=False):
        lines.append(f'\n{games.Data.iloc[0]} {games.Ora.iloc[0]} | {fixture}')
        for group in ('1X2', 'Under/Over 2.5'):
            markets = games[games.GruppoMercato == group]
            if markets.empty:
                continue
            cells = []
            for _, row in markets.iterrows():
                flag = ' [più probabile nel mercato]' if row.PiuProbabileNelMercato else ''
                cells.append(f'{row.Mercato} @ {row.QuotaSisal:.2f}: '
                             f'P {_prob(row.ProbMercatoNoVig)}, '
                             f'DC {_raw_probability(row.ProbPoissonDC)}{flag}')
            lines.append(f'  {group}: ' + ' | '.join(cells))
        if (games.QualitaStorico != 'OK').any():
            lines.append('  AVVISO: risultati storici da aggiornare; stime DC esplorative.')
        if (games.Stato == 'SQUADRE_NON_RICONOSCIUTE').any():
            lines.append('  AVVISO: squadra non identificata nello storico; DC non disponibile.')
    lines.append('\nNessuna indicazione di profitto o puntata: Sisal non validata fuori campione.')
    return '\n'.join(lines)


def write_readable_report(report: pd.DataFrame, destination: Path, *,
                          updated: object, now: object,
                          source_url: str, last_result: object,
                          freshness_reason: str, history_stale: bool,
                          unrecognized: list[str]) -> None:
    """Markdown report includes every quote and the top outcome per market."""
    lines = [
        '# Quote Sisal Serie A — probabilità e graduatoria',
        '',
        f'Foglio Sisal aggiornato: **{updated:%d/%m/%Y %H:%M}** (ora italiana). '
        f'Analisi: **{now:%d/%m/%Y %H:%M}**.',
        '',
        f'Fonte: {source_url}',
        '',
        '**Lettura:** P Sisal è una probabilità implicita ricavata dalle quote '
        '(margine ripartito proporzionalmente). NON è una previsione '
        'indipendente o una probabilità reale dimostrata.',
        '',
        '**DC** è la stima Poisson/Dixon–Coles, riportata per confronto: '
        'sul backtest disponibile è risultata meno precisa del mercato. '
        'Una differenza non dimostra valore atteso positivo.',
        '',
        f'Ultimo risultato storico: {last_result.date()}. {freshness_reason}',
        '',
    ]
    if history_stale:
        lines += ['**AVVISO: storico incompleto. Le stime DC sono esplorative; '
                  'le quote e P Sisal rimangono consultabili.**', '']
    lines += [
        '## Esiti più probabili per partita (secondo Sisal senza margine)',
        '',
        '| Data | Partita | Migliore 1X2 | P Sisal | Migliore U/O 2,5 | P Sisal |',
        '|---|---|---|---:|---|---:|',
    ]
    for (_, _, fixture), games in report.groupby(['Data', 'Ora', 'Partita'],sort=False):
        def best(group: str) -> tuple[str, str]:
            values = games[(games.GruppoMercato==group) & (games.RankMercato==1)]
            if values.empty:
                return ('—', '—')
            row=values.iloc[0]
            return (f'{row.Mercato} @ {row.QuotaSisal:.2f}', _prob(row.ProbMercatoNoVig))
        first, p_first = best('1X2')
        goals, p_goals = best('Under/Over 2.5')
        lines.append(f'| {games.Data.iloc[0]} {games.Ora.iloc[0]} | {fixture} | '
                     f'{first} | {p_first} | {goals} | {p_goals} |')

    lines += [
        '',
        '## Tutte le quote, ordinate per probabilità nel singolo mercato',
        '',
        '| Data | Partita | Mercato | Esito | Quota Sisal | P Sisal no-vig | DC | Posizione |',
        '|---|---|---|---|---:|---:|---:|---:|',
    ]
    for _, row in report.iterrows():
        lines.append(
            f'| {row.Data} {row.Ora} | {row.Partita} | {row.GruppoMercato} | '
            f'{row.Mercato} | {row.QuotaSisal:.2f} | '
            f'{_prob(row.ProbMercatoNoVig)} | '
            f'{_raw_probability(row.ProbPoissonDC)} | {int(row.RankMercato)} |'
        )
    if unrecognized:
        lines += ['', 'Squadre non riconosciute per il modello (ma con quote Sisal '
                  'comunque riportate): ' + '; '.join(sorted(set(unrecognized))) + '.']
    lines += [
        '',
        'Le quote possono cambiare. Nessuna delle percentuali è una garanzia. '
        '**Non sono proposte puntate**: non esiste ancora una validazione '
        'indipendente su uno storico di quote Sisal registrate prima delle partite.',
        '',
    ]
    destination.write_text('\n'.join(lines),encoding='utf-8')
