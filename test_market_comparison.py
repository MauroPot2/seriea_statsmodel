"""Deterministic protections against circular probabilities and fictitious odds."""
import tempfile
import unittest
from pathlib import Path
from datetime import timedelta

import pandas as pd

from market_comparison import (
    independent_consensus, comparative_prices, historical_books,
    compare_historical, read_live_snapshot, compare_live_snapshot, no_vig,
)


class TestHistoricalMarket(unittest.TestCase):
    def setUp(self):
        self.books={
            'A':{'1':2.0,'X':3.2,'2':4.0},
            'B':{'1':2.1,'X':3.1,'2':3.8},
            'C':{'1':2.05,'X':3.3,'2':3.7},
            'D':{'1':2.2,'X':3.2,'2':3.6},
        }

    def test_without_target_quote_even_if_extreme(self):
        first=independent_consensus(self.books,'A')
        alternative={**self.books,'A':{'1':100.0,'X':1.1,'2':1.2}}
        second=independent_consensus(alternative,'A')
        self.assertEqual(first,second)
        self.assertNotIn('A',first['source_books'])
        self.assertEqual(first['n_others'],3)

    def test_probability_sums_to_one(self):
        p=no_vig(self.books['B'])
        self.assertAlmostEqual(sum(p.values()),1,places=12)
        baseline=independent_consensus(self.books,'A')
        self.assertAlmostEqual(sum(baseline['probabilities'].values()),1,places=12)

    def test_must_have_at_least_three_other_books(self):
        fewer={k:v for k,v in self.books.items() if k in ('A','B','C')}
        offers=comparative_prices(fewer)
        self.assertEqual(len(offers),9)
        self.assertTrue(all(x['p_market_loo'] is None for x in offers))
        self.assertTrue(all(x['status']=='INSUFFICIENT_INDEPENDENT_BOOKS' for x in offers))

    def test_pinnacle_excluded_after_official_staleness_notice(self):
        row=pd.Series({'Date':pd.Timestamp('2025-08-31'),
            'B365H':2.0,'B365D':3.5,'B365A':4.1,
            'PSH':50.0,'PSD':50.0,'PSA':50.0,
            'BWH':2.1,'BWD':3.5,'BWA':3.8})
        prices=historical_books(row)
        self.assertNotIn('PS',prices)
        self.assertIn('B365',prices)
        early=row.copy()
        early['Date']=pd.Timestamp('2024-08-01')
        self.assertIn('PS',historical_books(early))

    def test_no_closing_or_aggregate_fallback(self):
        row=pd.Series({'Date':pd.Timestamp('2025-08-31'),
            'B365CH':2.0,'B365CD':3.5,'B365CA':4.0,
            'AvgH':2.0,'AvgD':3.5,'AvgA':4.0,
            'MaxH':2.3,'MaxD':4.0,'MaxA':5.5,
            'B365H':2.0,'B365D':None,'B365A':4.0})
        self.assertNotIn('B365',historical_books(row))
        self.assertFalse(historical_books(row))

    def test_historical_paired_comparison_never_reports_verified_profit(self):
        row=pd.Series({'Date':pd.Timestamp('2025-08-31'),
            'HomeTeam':'A','AwayTeam':'B','FTHG':2,'FTAG':1,
            'B365H':2.0,'B365D':3.2,'B365A':4.0,
            'BWH':2.1,'BWD':3.1,'BWA':3.8,
            'VCH':2.05,'VCD':3.3,'VCA':3.7,
            'WHH':2.2,'WHD':3.2,'WHA':3.6})
        with tempfile.TemporaryDirectory() as tmp:
            metrics=compare_historical(pd.DataFrame([row]),Path(tmp))
            self.assertEqual(metrics['n_matches_compared'],1)
            self.assertFalse(metrics['positive_return_validated'])
            saved=pd.read_csv(Path(tmp)/'market_historical_prices.csv')
            self.assertEqual(len(saved),12)
            self.assertIn('RETROSPECTIVE_UNSYNCHRONIZED',saved.status.unique())
            assert (Path(tmp)/'market_historical_report.md').exists()


class TestLiveMarket(unittest.TestCase):
    def setUp(self):
        now=pd.Timestamp.now(tz='UTC').floor('s')
        self.capture=now-pd.Timedelta(minutes=5)
        self.kickoff=now+pd.Timedelta(hours=8)
        data=[]
        for book,prices in [('Sisal',(2.05,3.35,4.1)),
                            ('A',(2.0,3.2,3.9)),
                            ('B',(2.1,3.3,3.85)),
                            ('C',(2.0,3.4,4.0))]:
            for sel,price in zip(('1','X','2'),prices):
                data.append({'captured_at':self.capture.isoformat(),
                             'kickoff':self.kickoff.isoformat(),
                             'home':'Milan','away':'Roma','bookmaker':book,
                             'market_group':'1X2','outcome':sel,'odds':price})
        self.df=pd.DataFrame(data)

    def load(self,rows):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'quotes.csv'
            rows.to_csv(path,index=False)
            return read_live_snapshot(path)

    def test_full_snapshot_preserves_every_source_and_outcome(self):
        frame=self.load(self.df)
        with tempfile.TemporaryDirectory() as temp:
            board,state=compare_live_snapshot(frame,Path(temp))
            self.assertEqual(len(board),12)
            self.assertEqual(state['prices'],12)
            self.assertEqual(state['excluded_groups'],[])
            self.assertFalse(state['independent_market_probability_validated'])
            self.assertIn('Sisal',board.bookmaker.values)
            self.assertTrue((board.n_others==3).all())

    def test_duplicate_prices_raise_error(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.load(pd.concat([self.df,self.df.iloc[:1]],ignore_index=True))

    def test_naive_timestamps_rejected(self):
        bad=self.df.copy()
        bad['captured_at']=bad.captured_at.str.replace(r'\+00:00$','',regex=True)
        with self.assertRaisesRegex(ValueError,'timezone'):
            self.load(bad)

    def test_no_post_kickoff_quotes(self):
        bad=self.df.copy()
        bad['captured_at']=(self.kickoff+pd.Timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(ValueError,'Post-kickoff'):
            self.load(bad)

    def test_incomplete_book_has_no_estimated_probability(self):
        bad=self.df[self.df.bookmaker!='C'].copy()
        frame=self.load(bad)
        with tempfile.TemporaryDirectory() as temp:
            board,_=compare_live_snapshot(frame,Path(temp))
            self.assertTrue(board.p_market_loo.isna().all())
            self.assertTrue((board.n_others==0).all())

    def test_large_capture_skew_rejected(self):
        bad=self.df.copy()
        bad.loc[bad.bookmaker=='C','captured_at'] = (
            self.capture-pd.Timedelta(minutes=9)).isoformat()
        frame=self.load(bad)
        with tempfile.TemporaryDirectory() as temp:
            board,state=compare_live_snapshot(frame,Path(temp))
            self.assertTrue(board.empty)
            self.assertEqual(len(state['excluded_groups']),1)

    def test_invalid_price_rejected(self):
        bad=self.df.copy()
        bad.loc[0,'odds']=0.95
        with self.assertRaisesRegex(ValueError,'Invalid or missing'):
            self.load(bad)


if __name__=='__main__':
    unittest.main()
