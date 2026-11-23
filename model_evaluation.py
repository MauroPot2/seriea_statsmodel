"""Pre-registered, chronological evaluation of Serie A probability forecasts.

Calibration is trained on the 2024/25 season only. Evaluation starts 2025/26;
we never optimise parameters/filters after observing this held-out period.
Opening Bet365 prices are a proxy benchmark; this is NOT Sisal validation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CALIBRATION_START = pd.Timestamp('2024-09-01')
HOLDOUT_START = pd.Timestamp('2025-08-01')
CALIBRATION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_CAL_MATCHES = 100
PROFILE_VERSION = 1
PRICING = 'Bet365_opening'
OUTCOMES = {'1X2': ('1', 'X', '2'), 'OU25': ('OVER2.5', 'UNDER2.5')}


def blend_probabilities(raw: dict[str, float], odds: dict[str, float],
                        weights: dict[str, float]) -> dict[str, float]:
    """Linear pool of Poisson-DC and vig-removed market probabilities."""
    from seriea_value_model import no_vig
    result = dict(raw)
    for group, keys in OUTCOMES.items():
        baseline = no_vig(odds, keys)
        if not baseline:
            continue
        weight = float(weights[group])
        if not 0.0 <= weight <= 1.0 or not np.isfinite(weight):
            raise ValueError('Invalid market calibration weight')
        for market in keys:
            result[market] = weight * raw[market] + (1.0-weight) * baseline[market]
    return result


def load_validation_profile(directory: Path) -> dict | None:
    p = directory / 'validation_profile.json'
    if not p.exists():
        return None
    try:
        profile = json.loads(p.read_text(encoding='utf-8'))
        if (profile.get('version') != PROFILE_VERSION or
            profile.get('price_source') != PRICING or
            profile.get('calibration_start') != str(CALIBRATION_START.date()) or
            profile.get('holdout_start') != str(HOLDOUT_START.date()) or
            profile.get('approved_for_sisal') is not False):
            return None
        weights = profile['weights']
        if set(weights) != set(OUTCOMES):
            return None
        if any(not 0 <= float(w) <= 1 or not np.isfinite(float(w)) for w in weights.values()):
            return None
        return profile
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def walk_forward(df: pd.DataFrame, start: pd.Timestamp,
                 end: pd.Timestamp | None = None) -> list[dict]:
    """One model per day fitted only on strictly earlier match results."""
    from seriea_value_model import (fit, odds_from_row, no_vig, strength_counts,
                                    MIN_HISTORY)
    selected = df[df.Date >= start]
    if end is not None:
        selected = selected[selected.Date < end]
    output: list[dict] = []
    for day in sorted(selected.Date.unique()):
        day = pd.Timestamp(day)
        past = df[df.Date < day]
        if len(past) < MIN_HISTORY:
            continue
        model = fit(past, day)
        for _, row in selected[selected.Date == day].iterrows():
            odds = odds_from_row(row, price_kind='opening')
            if not no_vig(odds, OUTCOMES['1X2']):
                continue
            raw = model.probabilities(str(row.HomeTeam), str(row.AwayTeam))
            independent = model.probabilities(str(row.HomeTeam), str(row.AwayTeam), dc=False)
            expected = {}
            for keys in OUTCOMES.values():
                expected.update(no_vig(odds, keys))
            n_home, n_away = strength_counts(past, str(row.HomeTeam), str(row.AwayTeam))
            hg, ag = int(row.FTHG), int(row.FTAG)
            winner = '1' if hg > ag else ('X' if hg == ag else '2')
            total = 'OVER2.5' if hg+ag >= 3 else 'UNDER2.5'
            output.append({'date': day, 'home': row.HomeTeam, 'away': row.AwayTeam,
                           'winner': winner, 'total': total,
                           'odds': odds, 'market': expected, 'raw': raw,
                           'independent': independent,
                           'team_history_min': min(n_home, n_away)})
    return output


def multiclass_metrics(events: list[dict], group: str,
                       method: str, weights: dict[str, float] | None = None) -> dict:
    keys = OUTCOMES[group]
    scores = []
    briers = []
    for event in events:
        if any(k not in event['market'] for k in keys):
            continue
        if method == 'blended':
            probs = blend_probabilities(event['raw'], event['odds'], weights)
        else:
            probs = event[method]
        true = event['winner'] if group == '1X2' else event['total']
        p = float(probs[true])
        vec = np.array([probs[k] for k in keys], dtype=float)
        actual = np.array([float(k == true) for k in keys])
        if p <= 0 or p > 1 or not np.all(np.isfinite(vec)):
            raise ValueError('Invalid probability in evaluation')
        scores.append(-math.log(p))
        briers.append(float(np.square(vec-actual).sum()))
    if not scores:
        return {'n': 0, 'log_loss': None, 'brier': None}
    return {'n': len(scores), 'log_loss': float(np.mean(scores)),
            'brier': float(np.mean(briers))}


def select_pool_weights(calibration: list[dict]) -> tuple[dict[str, float], dict]:
    """Pre-holdout log-loss-only selection; NEVER select using test ROI."""
    weights = {}
    details = {}
    for group in OUTCOMES:
        base = multiclass_metrics(calibration, group, 'market')
        if base['n'] < MIN_CAL_MATCHES:
            weights[group] = 0.0
            details[group] = {'n': base['n'], 'weight': 0.0,
                              'note': 'INSUFFICIENT_CALIBRATION_SAMPLE'}
            continue
        losses = []
        for w in CALIBRATION_WEIGHTS:
            trial = {'1X2': w, 'OU25': w}
            m = multiclass_metrics(calibration, group, 'blended', trial)
            losses.append((m['log_loss'], w))
        best_loss, best_weight = min(losses, key=lambda pair: (pair[0], pair[1]))
        weights[group] = best_weight
        details[group] = {'n': base['n'], 'weight': best_weight,
                          'calibration_log_loss': best_loss,
                          'market_log_loss': base['log_loss'],
                          'note': 'WEIGHT_LOCKED_BEFORE_HOLDOUT'}
    return weights, details


def price_bucket(price: float) -> str:
    if price < 1.7: return '<1.70'
    if price < 2.5: return '1.70-2.49'
    if price < 4.0: return '2.50-3.99'
    return '>=4.00'


def reliability_bins(events: list[dict], weights: dict[str, float]) -> pd.DataFrame:
    records = []
    for event in events:
        p = blend_probabilities(event['raw'], event['odds'], weights)
        for group, keys in OUTCOMES.items():
            if any(k not in event['market'] for k in keys):
                continue
            target = event['winner'] if group == '1X2' else event['total']
            for outcome in keys:
                prob = float(p[outcome])
                records.append({'market_group': group, 'outcome': outcome,
                                'bin': min(9, int(prob * 10)),
                                'prob': prob, 'observed': int(outcome == target)})
    columns = ['market_group', 'outcome', 'bin_start', 'bin_end', 'n', 'avg_prediction',
               'observed_frequency', 'absolute_calibration_gap']
    if not records: return pd.DataFrame(columns=columns)
    rows = []
    for (group, outcome, bucket), frame in pd.DataFrame(records).groupby(
            ['market_group', 'outcome', 'bin'], sort=True):
        predicted = frame.prob.mean()
        observed = frame.observed.mean()
        rows.append({'market_group': group, 'outcome': outcome,
                     'bin_start': bucket/10.0, 'bin_end': (bucket+1)/10.0,
                     'n': len(frame), 'avg_prediction': predicted,
                     'observed_frequency': observed,
                     'absolute_calibration_gap': abs(predicted-observed)})
    return pd.DataFrame(rows, columns=columns)


def simulate_bets(events: list[dict], weights: dict[str, float],
                  bankroll: float) -> tuple[pd.DataFrame, dict]:
    """Paper-only simulations, never real Sisal stakes."""
    from seriea_value_model import qualify, MIN_TEAM_MATCHES
    capital, peak, drawdown = bankroll, bankroll, 0.0
    rows = []
    for event in events:
        if event['team_history_min'] < MIN_TEAM_MATCHES: continue
        probs = blend_probabilities(event['raw'], event['odds'], weights)
        choices = qualify(probs, event['odds'])
        if not choices or capital <= 0: continue
        pick = choices[0]
        stake = capital * pick['fraction']
        if stake < 0.01: continue
        winning = pick['market'] in (event['winner'], event['total'])
        pnl = stake*(pick['odds']-1) if winning else -stake
        capital += pnl
        peak = max(peak, capital)
        drawdown = max(drawdown, (peak-capital)/peak)
        rows.append({'date': str(event['date'].date()), 'home': event['home'],
                     'away': event['away'], 'market': pick['market'],
                     'price_bucket': price_bucket(pick['odds']),
                     'odds': pick['odds'], 'prob': pick['prob'],
                     'ev_estimated': pick['ev'], 'edge_estimated': pick['edge'],
                     'stake': stake, 'won': winning, 'profit': pnl,
                     'capital': capital})
    frame = pd.DataFrame(rows, columns=['date', 'home', 'away', 'market',
                      'price_bucket', 'odds', 'prob', 'ev_estimated',
                      'edge_estimated', 'stake', 'won', 'profit', 'capital'])
    total_staked = float(frame.stake.sum()) if len(frame) else 0.0
    roi = float(frame.profit.sum()/total_staked) if total_staked else None
    interval = None
    if len(frame) >= 30 and frame.date.nunique() >= 20:
        grouped = frame.groupby('date')[['profit','stake']].sum().to_numpy()
        rng = np.random.default_rng(2026)
        indices = rng.integers(len(grouped), size=(3000, len(grouped)))
        sampled = grouped[indices].sum(axis=1)
        ratios = sampled[:,0] / np.maximum(sampled[:,1], 1e-9)
        interval = [float(v) for v in np.quantile(ratios, [0.025,0.975])]
    return frame, {'n_bets': int(len(frame)), 'total_staked': total_staked,
                   'profit': float(frame.profit.sum()) if len(frame) else 0.0,
                   'roi': roi, 'final_bankroll': float(capital),
                   'max_drawdown': float(drawdown),
                   'bootstrap_day_95_roi': interval}


def run_validation(df: pd.DataFrame, start: pd.Timestamp,
                   bankroll: float, directory: Path) -> None:
    """Freeze calibration on 2024/25 and test 2025/26+ without score tuning."""
    start = pd.Timestamp(start).normalize()
    if start < HOLDOUT_START:
        raise ValueError(f'Test cannot start before reserved holdout {HOLDOUT_START.date()}')
    if df.Date.max() < start:
        raise ValueError('No results available after holdout start')
    print('Calibration walk-forward: 2024-09-01 to 2025-07-31, NO holdout data')
    calibr = walk_forward(df, CALIBRATION_START, HOLDOUT_START)
    weights, tuning = select_pool_weights(calibr)
    print('Locked linear-pool weights (1.0 means Poisson-DC; 0.0 market only): '
          f'1X2={weights["1X2"]:.2f}, OU25={weights["OU25"]:.2f}')
    print(f'Holdout walk-forward: {start.date()} to {df.Date.max().date()}')
    events = walk_forward(df, start)
    if not events:
        raise ValueError('No holdout matches with full Bet365 opening 1X2 odds')
    all_metrics = {}
    for group in OUTCOMES:
        all_metrics[group] = {
            label: multiclass_metrics(events, group, label, weights)
            for label in ('independent', 'raw', 'market', 'blended')}
    bets, bet_metrics = simulate_bets(events, weights, bankroll)
    diagnostic = reliability_bins(events, weights)
    directory.mkdir(exist_ok=True, parents=True)
    diagnostic.to_csv(directory/'calibration_bins.csv', index=False)
    bets.to_csv(directory/'backtest_bets.csv', index=False)
    predictions = []
    for event in events:
        blended = blend_probabilities(event['raw'], event['odds'], weights)
        for group, keys in OUTCOMES.items():
            if any(k not in event['market'] for k in keys): continue
            target = event['winner'] if group == '1X2' else event['total']
            for market in keys:
                predictions.append({'date': str(event['date'].date()),
                                    'home': event['home'], 'away': event['away'],
                                    'group': group, 'market': market,
                                    'actual': int(target == market),
                                    'price': event['odds'][market],
                                    'p_poisson': event['independent'][market],
                                    'p_dixon_coles': event['raw'][market],
                                    'p_market_no_vig': event['market'][market],
                                    'p_calibrated': blended[market]})
    pd.DataFrame(predictions).to_csv(directory/'backtest_predictions.csv', index=False)
    segments = []
    if len(bets):
        for (group, name), sub in pd.concat([
                bets.assign(group='mercato', name=bets.market),
                bets.assign(group='fascia_quota', name=bets.price_bucket)]).groupby(
                    ['group', 'name'], sort=True):
            segments.append({'group': group, 'name': name, 'n': len(sub),
                             'stake': float(sub.stake.sum()),
                             'roi': float(sub.profit.sum()/sub.stake.sum())})
    segments_df = pd.DataFrame(segments, columns=['group', 'name', 'n', 'stake', 'roi'])
    segments_df.to_csv(directory/'backtest_segments.csv', index=False)
    ll_1x2 = all_metrics['1X2']
    score_beats_market = (
        all_metrics['1X2']['blended']['log_loss'] < ll_1x2['market']['log_loss'] and
        all_metrics['1X2']['blended']['brier'] < ll_1x2['market']['brier'])
    ci = bet_metrics['bootstrap_day_95_roi']
    positive_robust_roi = (ci is not None and ci[0] > 0)
    verified_against_bet365 = bool(score_beats_market and positive_robust_roi and
                                   len(events) >= 200 and len(bets) >= 100)
    profile = {'version': PROFILE_VERSION, 'price_source': PRICING,
               'calibration_start': str(CALIBRATION_START.date()),
               'holdout_start': str(HOLDOUT_START.date()),
               'test_start': str(start.date()),
               'test_end': str(df.Date.max().date()),
               'weights': weights, 'calibration_details': tuning,
               'holdout_matches': len(events), 'holdout_metrics': all_metrics,
               'paper_bets': bet_metrics, 'passed_bet365_gate': verified_against_bet365,
               'approved_for_sisal': False,
               'reason': 'Sisal historical pre-match quotes and timestamps not evaluated',
               'note': 'No positive EV / profit claim without independent Sisal history.'}
    (directory/'validation_profile.json').write_text(
        json.dumps(profile, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# Rapporto validazione cronologica Serie A', '',
             f'**Calibrazione:** {CALIBRATION_START.date()} – 2025-07-31; '
             f'**holdout:** {start.date()} – {df.Date.max().date()}.',
             '', f'**Quota storica:** {PRICING}. Nessuna quota Sisal storica usata.',
             '', f'Pesi congelati: 1X2 = {weights["1X2"]:.2f}; '
             f'Under/Over = {weights["OU25"]:.2f}.',
             '', '| Mercato | Metodo | N | Log loss ↓ | Brier ↓ |',
             '|---|---|---:|---:|---:|']
    pretty = {'independent':'Poisson','raw':'Poisson + Dixon-Coles',
              'market':'Mercato senza margine','blended':'Calibrato (pool)'}
    for group in OUTCOMES:
        for method, metrics in all_metrics[group].items():
            if metrics['n']:
                lines.append(f'| {group} | {pretty[method]} | {metrics["n"]} | '
                             f'{metrics["log_loss"]:.4f} | {metrics["brier"]:.4f} |')
    lines.extend(['', '## Simulazione su quote Bet365 (non Sisal)', '',
                  f'- Partite esaminate: **{len(events)}**',
                  f'- Selezioni: **{len(bets)}**',
                  f'- Capitale iniziale/finale: **{bankroll:.2f} / {bet_metrics["final_bankroll"]:.2f} EUR**',
                  f'- ROI: **{bet_metrics["roi"]:+.1%}**' if bet_metrics['roi'] is not None else '- ROI: n/d (nessuna selezione)',
                  f'- Max drawdown: **{bet_metrics["max_drawdown"]:.1%}**',
                  f'- Bootstrap per giornate, IC 95% ROI: **{ci[0]:+.1%} .. {ci[1]:+.1%}**' if ci is not None else '- IC bootstrap: non calcolabile',
                  '', '## Diagnostica per fascia quota e mercato', '',
                  '| Raggruppamento | Categoria | N | ROI |',
                  '|---|---|---:|---:|'])
    for _, rec in segments_df.iterrows():
        lines.append(f'| {rec.group} | {rec["name"]} | {int(rec.n)} | {rec.roi:+.1%} |')
    if segments_df.empty:
        lines.append('| — | Nessuna selezione | 0 | n/d |')
    lines += ['', '## Interpretazione e limiti', '',
              f'- Supera gate fissato su Bet365 (test esplorativo): **{verified_against_bet365}**.',
              '- **Non validato contro Sisal. Nessuna puntata proposta.**',
              '- Non confrontare direttamente questi numeri con il vecchio backtest: '
              'ora si usano **solo quote di apertura**, non chiusura.',
              '- I pesi sono selezionati esclusivamente sul 2024/25 con log loss; '
              'il 2025/26+ non influisce sulla loro scelta.',
              '- L\'IC bootstrap raggruppa per data, ma non corregge tutte le dipendenze.',
              '- Uno storico con prezzi di apertura non prova che il prezzo fosse ottenibile.',
              '- Le curve di calibrazione sono in calibration_bins.csv.', '']
    (directory/'validation_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'\nHOLDOUT N={len(events)}; quote {PRICING} (non Sisal)')
    for group in OUTCOMES:
        print(f'[{group}]')
        for label, metrics in all_metrics[group].items():
            if metrics['n']:
                print(f'  {pretty[label]}: n={metrics["n"]}, '
                      f'log loss={metrics["log_loss"]:.4f}, Brier={metrics["brier"]:.4f}')
    if bet_metrics['roi'] is not None:
        print(f'SIMULATED BET365: {len(bets)} bets, ROI {bet_metrics["roi"]:+.1%}, '
              f'profit {bet_metrics["profit"]:+.2f} EUR, '
              f'bankroll {bet_metrics["final_bankroll"]:.2f} EUR')
        print(f'Drawdown {bet_metrics["max_drawdown"]:.1%}; 95% day bootstrap ROI: {ci}'
              if ci is not None else 'Bootstrap CI unavailable (insufficient observations)')
    else:
        print('SIMULATED BET365: 0 qualifying paper bets, ROI undefined.')
    print('VERIFIED BET365 GATE:', 'PASS' if verified_against_bet365 else 'FAIL')
    print('SISAL STATUS: NOT VALIDATED; no money stakes authorised by this program.')
    print('Files: validation_report.md, validation_profile.json, '
          'backtest_predictions.csv, backtest_segments.csv, '
          'backtest_bets.csv, calibration_bins.csv')
