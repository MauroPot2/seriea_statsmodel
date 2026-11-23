"""Secondary result feed must fail closed on missing or contradictory data."""
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import pandas as pd

from results_sync import ResultSyncError, sync_results


class TestSupplementaryResults(TestCase):
    def setUp(self):
        # Four fixture rounds and 20 distinct teams; first three have finished.
        teams = [f'Club{i}' for i in range(20)]
        results = []
        for gw in range(1, 4):
            for i in range(10):
                results.append({'gw': str(gw), 'date': f'2026-11-0{gw}',
                                'home': teams[i], 'away': teams[i+10],
                                'ft_home': str(gw), 'ft_away': '1'})
        future = [{'gw': '4', 'date': '2026-11-24', 'time': '20:45',
                   'home': teams[i], 'away': teams[i+10]} for i in range(10)]
        self.results = pd.DataFrame(results)
        self.fixtures = pd.DataFrame(future)
        self.primary = pd.DataFrame([
            {'Date': pd.Timestamp(row['date']), 'HomeTeam': row['home'],
             'AwayTeam': row['away'], 'FTHG': int(row['ft_home']),
             'FTAG': int(row['ft_away'])} for row in results[:-1]])

    def sync(self):
        with tempfile.TemporaryDirectory() as folder:
            return sync_results(self.primary, datetime(2026, 11, 23, 9, 14),
                                Path(folder), results=self.results,
                                fixtures=self.fixtures)

    def test_missing_result_is_added_once(self):
        merged, reason = self.sync()
        self.assertEqual(len(merged), len(self.primary)+1)
        self.assertIn('1 integrati', reason)

    def test_conflicting_score_blocks_sync(self):
        self.results.loc[0, 'ft_home'] = '9'
        with self.assertRaisesRegex(ResultSyncError, 'Conflitto'):
            self.sync()

    def test_overdue_fixture_blocks_sync(self):
        self.fixtures.loc[0, 'date'] = '2026-11-22'
        with self.assertRaisesRegex(ResultSyncError, 'già iniziata'):
            self.sync()

    def test_missing_fixture_blocks_sync(self):
        self.fixtures = self.fixtures.iloc[:-1]
        with self.assertRaisesRegex(ResultSyncError, 'diverso da 10'):
            self.sync()

    def test_unknown_team_blocks_sync(self):
        self.results.loc[0, 'home'] = 'ClubSconosciuto'
        with self.assertRaisesRegex(ResultSyncError, 'non riconosciute'):
            self.sync()
