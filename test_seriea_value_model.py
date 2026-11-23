"""Run with: python -m unittest test_seriea_value_model.py"""
import unittest
import numpy as np
import pandas as pd

from seriea_value_model import Model, no_vig, qualify, odds_from_row

class TestSerieAModel(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        m = Model(['A','B'],np.array([0.15,-0.15]),np.array([-0.12,0.12]),0.3,0.04)
        p = m.probabilities('A','B')
        self.assertAlmostEqual(sum(p[k] for k in ('1','X','2')),1,places=11)
        self.assertAlmostEqual(p['OVER2.5']+p['UNDER2.5'],1,places=11)

    def test_no_vig(self):
        q = no_vig({'1':2.0,'X':3.3,'2':4.0},('1','X','2'))
        self.assertAlmostEqual(sum(q.values()),1,places=12)

    def test_no_positive_ev_no_bet(self):
        probs={'1':.5,'X':.3,'2':.2,'OVER2.5':.5,'UNDER2.5':.5}
        odds={'1':1.8,'X':3.0,'2':4.5,'OVER2.5':1.8,'UNDER2.5':1.8}
        self.assertEqual(qualify(probs,odds),[])

    def test_consistent_market_time(self):
        row=pd.Series({'B365CH':2.2,'B365CD':3.3,'B365CA':float('nan'),
                       'B365H':2.1,'B365D':3.2,'B365A':3.8})
        self.assertEqual(odds_from_row(row),{'1':2.1,'X':3.2,'2':3.8})

if __name__ == '__main__':
    unittest.main()
