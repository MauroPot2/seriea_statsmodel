"""Deterministic tests for calibration, time splits and paper-only betting."""
import math
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from seriea_value_model import Model, odds_from_row
from model_evaluation import (blend_probabilities, select_pool_weights,
                              walk_forward, simulate_bets, load_validation_profile,
                              reliability_bins, HOLDOUT_START)


class TestModelEvaluation(TestCase):
    def setUp(self):
        self.odds={'1':2.2,'X':3.3,'2':3.3,'OVER2.5':2.0,'UNDER2.5':1.95}
        self.raw={'1':.5,'X':.25,'2':.25,'OVER2.5':.58,'UNDER2.5':.42}
        self.zero={'1X2':0.,'OU25':0.}

    def fixture(self):
        return {'date':pd.Timestamp('2025-09-01'),'home':'A','away':'B',
                'winner':'1','total':'OVER2.5','odds':self.odds,
                'market':{'1':.4,'X':.3,'2':.3,'OVER2.5':.49,'UNDER2.5':.51},
                'raw':self.raw,'independent':self.raw,'team_history_min':15}

    def test_dc_probability_distribution(self):
        m=Model(['A','B'],np.array([0.,0.]),np.array([0.,0.]),
                math.log(1.5),math.log(1.2),rho=-.08)
        dc=m.probabilities('A','B')
        pois=m.probabilities('A','B',dc=False)
        self.assertAlmostEqual(dc['1']+dc['X']+dc['2'],1)
        self.assertAlmostEqual(dc['OVER2.5']+dc['UNDER2.5'],1)
        self.assertNotAlmostEqual(dc['X'],pois['X'],places=5)

    def test_market_anchor_avoids_positive_ev_claims(self):
        from seriea_value_model import qualify, no_vig
        p=blend_probabilities(self.raw,self.odds,self.zero)
        self.assertEqual(qualify(p,self.odds),[])
        for keys in [('1','X','2'),('OVER2.5','UNDER2.5')]:
            expected=no_vig(self.odds,keys)
            for outcome in keys:
                self.assertAlmostEqual(p[outcome],expected[outcome])

    def test_blended_probabilities_normalised(self):
        p=blend_probabilities(self.raw,self.odds,{'1X2':.25,'OU25':.75})
        self.assertAlmostEqual(p['1']+p['X']+p['2'],1)
        self.assertAlmostEqual(p['OVER2.5']+p['UNDER2.5'],1)

    def test_calibration_small_sample_defaults_to_market(self):
        weights,info=select_pool_weights([self.fixture()])
        self.assertEqual(weights,self.zero)
        self.assertEqual(info['1X2']['note'],'INSUFFICIENT_CALIBRATION_SAMPLE')

    def test_no_paper_bet_when_shrunk_to_market(self):
        frame,stats=simulate_bets([self.fixture()],self.zero,60.)
        self.assertEqual(len(frame),0)
        self.assertIsNone(stats['roi'])
        self.assertEqual(stats['final_bankroll'],60.)

    def test_opening_prices_never_substitute_closing(self):
        row=pd.Series({'B365CH':2.2,'B365CD':3.3,'B365CA':4.4})
        self.assertEqual(odds_from_row(row),{})
        self.assertEqual(odds_from_row(row,price_kind='closing'),
                         {'1':2.2,'X':3.3,'2':4.4})

    def test_walkforward_has_no_results_from_current_day(self):
        frame=pd.DataFrame([{'Date':pd.Timestamp('2025-07-01'),
                            'HomeTeam':'A','AwayTeam':'B','FTHG':1,'FTAG':0}]*300
             +[{'Date':pd.Timestamp('2025-08-01'),'HomeTeam':'A','AwayTeam':'B',
                'FTHG':2,'FTAG':1,'B365H':2.2,'B365D':3.3,'B365A':3.3},
               {'Date':pd.Timestamp('2025-08-02'),'HomeTeam':'A','AwayTeam':'B',
                'FTHG':3,'FTAG':1,'B365H':2.2,'B365D':3.3,'B365A':3.3}])
        model=Model(['A','B'],np.zeros(2),np.zeros(2),.3,.1)
        cutoffs=[]
        def strict_fit(past,as_of):
            self.assertTrue((past.Date<as_of).all())
            cutoffs.append(pd.Timestamp(as_of))
            return model
        with patch('seriea_value_model.fit',side_effect=strict_fit):
            rows=walk_forward(frame,HOLDOUT_START)
        self.assertEqual(len(rows),2)
        self.assertEqual(cutoffs,[pd.Timestamp('2025-08-01'),pd.Timestamp('2025-08-02')])

    def test_reliability_bins_report_outcomes(self):
        frame=reliability_bins([self.fixture()],{'1X2':1.,'OU25':1.})
        self.assertEqual(int(frame.n.sum()),5)
        self.assertTrue(frame.observed_frequency.between(0,1).all())

    def test_missing_profile_does_not_authorise_sisal(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(load_validation_profile(Path(td)))
