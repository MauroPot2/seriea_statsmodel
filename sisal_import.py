"""Importer for Sisal's publicly downloadable *Calcio Base per Manifestazione* PDF.

Only reads public pre-match prices; never logs in, places bets or bypasses controls.
PDFs are not APIs and layouts may change. Fail closed rather than guessing columns.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
import re
import urllib.request

import fitz  # PyMuPDF

OFFICIAL_PDF_URL = (
    'https://landing.sisal.it/volantini/Scommesse_Sport/Quote/'
    'calcio%20base%20per%20manifestazione.pdf'
)

# Column edges measured against the published *landscape* Sisal PDF layout.
# Each pair is a fraction of the page width; use centers of PDF text words.
COLUMNS = {
    'Date': (0.020, 0.060),
    'Time': (0.060, 0.098),
    'HomeTeam': (0.174, 0.297),
    'AwayTeam': (0.297, 0.419),
    'OddsH': (0.437, 0.464),
    'OddsD': (0.464, 0.488),
    'OddsA': (0.488, 0.516),
    'OddsUnder25': (0.727, 0.751),
    'OddsOver25': (0.751, 0.777),
}
NUMBER = re.compile(r'^\d{1,3}[.,]\d{1,2}$')
DAY_MONTH = re.compile(r'^\d{1,2}/\d{1,2}$')
FOOTER = re.compile(
    r'[Dd]ati\s+aggiornati\s+al\s+(\d{1,2}/\d{1,2}/\d{4})'
    r'\s*[-–]?\s*ore\s*(\d{1,2}[:.]\d{2})', re.IGNORECASE
)


class SisalImportError(ValueError):
    """Malformed, unavailable or stale quote sheet."""


def download_sisal_pdf(destination: Path, url: str = OFFICIAL_PDF_URL) -> bytes:
    """Download the public PDF; validate before saving it."""
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; SisalQuoteResearch/1.0)',
        'Accept': 'application/pdf',
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = response.read(15_000_001)
    except Exception as exc:
        raise SisalImportError(
            'Impossibile scaricare il PDF Sisal. Scaricalo dal foglio quote ufficiale '
            'e riprova con: --pdf /percorso/tuo-file.pdf'
        ) from exc
    if not payload.startswith(b'%PDF-') or len(payload) > 15_000_000:
        raise SisalImportError('La risposta Sisal non è un PDF valido (oppure supera 15 MB).')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return payload


def pdf_words_on_line(words: list[tuple], baseline: float, tolerance: float = 3.1) -> list[tuple]:
    return sorted((w for w in words if abs((w[1]+w[3])/2-baseline) <= tolerance),
                  key=lambda w: w[0])


def cell_text(words: list[tuple], page_width: float, column: str) -> str:
    x0, x1 = COLUMNS[column]
    segment = [w for w in words if x0 <= (w[0]+w[2])/(2*page_width) < x1]
    return ' '.join(str(w[4]) for w in sorted(segment,key=lambda w:w[0])).strip()


def number_from_cell(raw: str) -> float | None:
    val = raw.strip().replace(' ', '')
    if not NUMBER.fullmatch(val):
        return None
    out = float(val.replace(',', '.'))
    return out if 1.001 < out < 250 else None


def infer_match_date(day_month: str, updated_date: date) -> date:
    d, m = (int(x) for x in day_month.split('/'))
    dates = []
    for year in (updated_date.year-1, updated_date.year, updated_date.year+1):
        try:
            candidate = date(year, m, d)
        except ValueError:
            continue
        if (candidate-updated_date).days >= -2:
            dates.append(candidate)
    if not dates:
        raise SisalImportError(f'Data partita non coerente col foglio: {day_month}')
    return min(dates, key=lambda candidate: (candidate - updated_date).days)


def find_pdf_update(pages: Any) -> datetime | None:
    updates = []
    for page in pages:
        match = FOOTER.search(page.get_text('text', sort=True))
        if match:
            d = datetime.strptime(match.group(1), '%d/%m/%Y').date()
            h = match.group(2).replace('.', ':')
            hh, mm = (int(x) for x in h.split(':'))
            updates.append(datetime.combine(d,time(hh,mm)))
    return max(updates) if updates else None


def section_at_line(words: list[tuple], y: float) -> str | None:
    line = pdf_words_on_line(words,y,tolerance=3.5)
    head = ' '.join(str(w[4]) for w in line if w[0] < 0.40*max(z[2] for z in words))
    head = re.sub(r'\s+', ' ', head).strip()
    if re.search(r'\bITA\s+Serie\s+A\b',head,re.IGNORECASE):
        return 'SERIE_A'
    # A new section ends Serie A, even when it is in another country.
    if re.match(r'^(ITA|INT|ENG|GER|SPA|FRA|POR|NED|ARG|BRA|USA)\s+\D',head,re.I):
        return 'OTHER'
    return None


def parse_sisal_pdf(payload: bytes, *, reference_date: date | None = None) -> tuple[list[dict], datetime]:
    """Return recognized Serie A fixtures and publication time.

    A row with ambiguous/missing 1X2 odds is rejected. An optional 2.5 pair is
    omitted when incomplete; the corresponding market is then not evaluated.
    """
    if not payload.startswith(b'%PDF-'):
        raise SisalImportError('Il file indicato non è un PDF.')
    try:
        doc = fitz.open(stream=payload, filetype='pdf')
    except Exception as exc:
        raise SisalImportError('Il PDF Sisal non può essere aperto.') from exc
    try:
        if doc.page_count > 150 or doc.page_count == 0:
            raise SisalImportError('Numero pagine del PDF non plausibile.')
        updated = find_pdf_update(doc)
        if updated is None:
            raise SisalImportError('Non trovo la data di aggiornamento nel footer Sisal; interrompo per sicurezza.')
        if reference_date is not None and (updated.date()-reference_date).days > 2:
            raise SisalImportError('Data di aggiornamento del PDF incompatibile con la data di analisi.')
        fixtures: list[dict] = []
        active = False
        for page in doc:
            words = page.get_text('words')
            width = float(page.rect.width)
            if width <= 0:
                continue
            # Include section markers and dated event rows in vertical reading order.
            anchors = []
            for w in words:
                value = str(w[4])
                yc = (w[1]+w[3])/2
                xpos = (w[0]+w[2])/(2*width)
                if value.upper() in ('ITA', 'INT', 'ENG', 'GER', 'SPA', 'FRA', 'POR', 'NED', 'ARG', 'BRA', 'USA') and xpos < .10:
                    anchors.append((yc, 'section'))
                elif DAY_MONTH.fullmatch(value) and xpos < .061:
                    anchors.append((yc, 'match'))
            for yc, kind in sorted(anchors,key=lambda item:item[0]):
                if kind == 'section':
                    section = section_at_line(words,yc)
                    if section:
                        active = section == 'SERIE_A'
                    continue
                if not active:
                    continue
                line = pdf_words_on_line(words,yc)
                date_str = cell_text(line,width,'Date')
                time_str = cell_text(line,width,'Time')
                home = cell_text(line,width,'HomeTeam')
                away = cell_text(line,width,'AwayTeam')
                if not DAY_MONTH.fullmatch(date_str) or not home or not away:
                    continue
                try:
                    match_date = infer_match_date(date_str,updated.date())
                except SisalImportError:
                    continue
                if not re.fullmatch(r'\d{1,2}[.:]\d{2}',time_str):
                    continue
                hh,mm=(int(v) for v in time_str.replace('.',':').split(':'))
                if hh>23 or mm>59:
                    continue
                result = {
                    'Date':match_date.isoformat(),
                    'Time':f'{hh:02d}:{mm:02d}',
                    'HomeTeamSisal':home,
                    'AwayTeamSisal':away,
                    'OddsH':number_from_cell(cell_text(line,width,'OddsH')),
                    'OddsD':number_from_cell(cell_text(line,width,'OddsD')),
                    'OddsA':number_from_cell(cell_text(line,width,'OddsA')),
                    'OddsUnder25':number_from_cell(cell_text(line,width,'OddsUnder25')),
                    'OddsOver25':number_from_cell(cell_text(line,width,'OddsOver25')),
                    'SisalUpdated':updated.isoformat(timespec='minutes'),
                }
                odds = [result[x] for x in ('OddsH','OddsD','OddsA')]
                if any(o is None for o in odds):
                    continue
                overround = sum(1/o for o in odds)
                if not 0.98 <= overround <= 1.50:
                    continue
                ou = [result['OddsUnder25'],result['OddsOver25']]
                if any(o is None for o in ou) or not 0.98 <= sum(1/o for o in ou) <= 1.50:
                    result['OddsUnder25'] = result['OddsOver25'] = None
                fixtures.append(result)
        dedup = {(r['Date'],r['Time'],r['HomeTeamSisal'],r['AwayTeamSisal']):r
                 for r in fixtures}
        rows = sorted(dedup.values(),key=lambda r:(r['Date'],r['Time'],r['HomeTeamSisal']))
        if not rows:
            raise SisalImportError(
                'Non riesco a estrarre incontri Serie A con quote 1X2 valide: '
                'potrebbe essere cambiata la struttura del PDF Sisal.'
            )
        return rows, updated
    finally:
        doc.close()


def check_freshness(updated: datetime, now: datetime, max_age_hours: float = 48) -> None:
    # Both timestamps are local Italian wall-clock times.
    age = (now - updated).total_seconds()/3600.0
    if age > max_age_hours:
        raise SisalImportError(
            f'Foglio Sisal vecchio di {age:.0f} ore (aggiornato {updated:%d/%m/%Y %H:%M}); '
            'non considero affidabili queste quote. Prova con un PDF aggiornato.'
        )
    if age < -5:
        raise SisalImportError('Data futura nel foglio Sisal: impossibile verificarne l’attualità.')
