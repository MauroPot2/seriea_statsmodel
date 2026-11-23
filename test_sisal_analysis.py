"""Regression tests for stale Football-Data results with a valid Sisal sheet."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import io
import math
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from seriea_value_model import Model, sisal_analysis, history_freshness
from test_sisal_import import fake_pdf


class TestSisalAnalysis(TestCase):
    def run_analysis(self, latest_result):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            pdf_path = base / 'foglio.pdf'
            pdf_path.write_bytes(fake_pdf())
            rows = []
            for i in range(320):
                home, away = [('Sassuolo', 'Napoli'), ('Cagliari', 'Milan')][i % 2]
                rows.append({'Date': pd.Timestamp(latest_result),
                             'HomeTeam': home, 'AwayTeam': away,
                             'FTHG': 1, 'FTAG': 1})
            history = pd.DataFrame(rows)
            names = ['Sassuolo', 'Napoli', 'Cagliari', 'Milan']
            model = Model(names, np.zeros(4), np.zeros(4), math.log(1.5), math.log(1.0))
            out, err = io.StringIO(), io.StringIO()
            with patch('seriea_value_model.read_history', return_value=history), \
                 patch('seriea_value_model.fit', return_value=model), \
                 redirect_stdout(out), redirect_stderr(err):
                sisal_analysis(pdf_path, base, 60.0, no_refresh=True,
                               analysis_time=datetime(2026, 11, 23, 9, 14))
            return (pd.read_csv(base / 'sisal_analisi.csv'),
                    (base / 'sisal_report.md').read_text(encoding='utf-8'),
                    out.getvalue() + err.getvalue())

    def test_outdated_history_is_research_only(self):
        frame, report, output = self.run_analysis('2026-11-08')
        self.assertGreater(len(frame), 0)
        self.assertTrue((frame.Stato == 'STORICO_DA_AGGIORNARE').all())
        self.assertTrue((frame.PuntataTeorica == 0).all())
        self.assertIn('AVVISO: storico incompleto', report)
        self.assertIn('quote analizzate', output)

    def test_recent_history_still_requires_independent_sisal_validation(self):
        frame, report, output = self.run_analysis('2026-11-22')
        self.assertGreater(len(frame), 0)
        self.assertTrue((frame.QualitaStorico == 'OK').all())
        self.assertTrue((frame.Stato == 'NON_VALIDATO_SISAL').all())
        self.assertTrue((frame.PuntataTeorica == 0).all())
        self.assertIn('tutti gli esiti disponibili', output)
        self.assertIn('Tutte le quote', report)


class TestInternationalBreakFreshness(TestCase):
    def test_november_break_is_not_stale_before_restart(self):
        fresh, reason = history_freshness(pd.Timestamp('2026-11-08'),
                                           datetime(2026, 11, 20, 20, 0))
        self.assertTrue(fresh)
        self.assertIn('Sosta nazionale', reason)

    def test_after_restart_missing_results_are_stale(self):
        fresh, _ = history_freshness(pd.Timestamp('2026-11-08'),
                                     datetime(2026, 11, 23, 9, 14))
        self.assertFalse(fresh)

    def test_old_unverified_gap_stays_stale(self):
        fresh, _ = history_freshness(pd.Timestamp('2026-09-20'),
                                     datetime(2026, 10, 8, 9, 0))
        self.assertFalse(fresh)
