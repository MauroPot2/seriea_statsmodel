"""xG provenance, date leakage and independent-probability regression tests."""
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

from xg_elo_model import (XGDataError, load_xg, forecast_asof,
                          elo_ratings, _score_probs, PROVIDERS)


class TestXGProvenance(TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir=Path(self.tmp.name)
        self.official=pd.DataFrame([
            dict(Date=pd.Timestamp('2025-09-01'),HomeTeam='Milan',
                 AwayTeam='Parma',FTHG=2,FTAG=1),
            dict(Date=pd.Timestamp('2025-09-02'),HomeTeam='Parma',
                 AwayTeam='Milan',FTHG=0,FTAG=0)])
        self.xg=pd.DataFrame([
            dict(date='2025-09-01',home='AC Milan',away='Parma Calcio 1913',
                 score='2-1',home_xg=1.5,away_xg=.8,
                 totals_unreliable=False),
            dict(date='2025-09-02',home='Parma Calcio 1913',away='AC Milan',
                 score='0–0',home_xg=.4,away_xg=1.3,
                 totals_unreliable=False)])
        self.target=self.dir/'xg_seriea_2025_understat.csv'

    def test_matching_source_aliases_and_scores(self):
        self.xg.to_csv(self.target,index=False)
        cleaned,detail=load_xg(self.dir,self.official)
        self.assertEqual(len(cleaned),2)
        self.assertEqual(detail['matched']['2025'],2)
        self.assertTrue((cleaned.Provider=='Understat').all())
        self.assertEqual(set(cleaned.HomeTeam),{'Milan','Parma'})

    def test_result_conflict_blocks_ingestion(self):
        self.xg.loc[0,'score']='3-1'
        self.xg.to_csv(self.target,index=False)
        with self.assertRaisesRegex(XGDataError,'conflict'):
            load_xg(self.dir,self.official)

    def test_missing_xg_is_not_imputed_with_goals(self):
        self.xg.loc[0,'home_xg']=np.nan
        self.xg.to_csv(self.target,index=False)
        cleaned,detail=load_xg(self.dir,self.official)
        self.assertEqual(len(cleaned),1)
        self.assertEqual(detail['ignored_missing_xg']['2025'],1)

    def test_2026_sportmonks_is_not_relabelled_understat(self):
        bad=self.dir/'xg_seriea_2026_understat.csv'
        self.xg.to_csv(bad,index=False)
        with self.assertRaisesRegex(XGDataError,'Unexpected source'):
            load_xg(self.dir,self.official)


class TestIndependentModels(TestCase):
    def setUp(self):
        # Historical sample; all teams have dated matches and xG in one season.
        teams=['A','B','C','D']
        rows=[]
        measured=[]
        for i in range(330):
            d=pd.Timestamp('2025-08-01')+pd.Timedelta(days=i//3)
            h,a=teams[i%4],teams[(i+1)%4]
            rows.append(dict(Date=d,HomeTeam=h,AwayTeam=a,FTHG=2,FTAG=1))
            measured.append(dict(Date=d,HomeTeam=h,AwayTeam=a,
                                 XGH=1.75,XGA=1.05,Provider='Understat',
                                 Season=2025))
        self.history=pd.DataFrame(rows)
        self.xg=pd.DataFrame(measured)

    def test_probabilities_normalize_in_both_markets(self):
        probs=_score_probs(1.9,1.0)
        self.assertAlmostEqual(sum(probs[k] for k in ('1','X','2')),1.)
        self.assertAlmostEqual(probs['OVER2.5']+probs['UNDER2.5'],1.)

    def test_elo_is_updated_without_same_day_order_dependency(self):
        sample=self.history.iloc[:2].copy()
        outcome1=elo_ratings(sample)
        outcome2=elo_ratings(sample.iloc[::-1])
        self.assertEqual(outcome1,outcome2)

    def test_forecast_excludes_same_day_xg_even_if_larger(self):
        cutoff=pd.Timestamp('2025-12-01')
        first=forecast_asof(self.history,self.xg,cutoff,'A','B')
        fake_future=self.xg.iloc[:4].copy()
        fake_future['Date']=cutoff
        fake_future['XGH']=11.0
        changed=pd.concat([self.xg,fake_future],ignore_index=True)
        second=forecast_asof(self.history,changed,cutoff,'A','B')
        self.assertEqual(first['status'],second['status'])
        self.assertEqual(first['hybrid'],second['hybrid'])

    def test_no_cross_provider_xg_used_as_current_form(self):
        cutoff=pd.Timestamp('2026-11-23')
        current=self.xg.copy()
        current['Season']=2026
        current['Provider']='Understat'  # invalid for 2026
        f=forecast_asof(self.history,current,cutoff,'A','B')
        self.assertEqual(f['status'],'ELO_ONLY')
        self.assertIsNone(f['xg'])

    def test_forecast_independent_of_market_prices(self):
        fc=forecast_asof(self.history,self.xg,pd.Timestamp('2025-12-01'),'A','B')
        self.assertIn(fc['status'],('ELO_ONLY','XG_ELO_AVAILABLE'))
        self.assertEqual(set(fc['elo']),{'1','X','2','OVER2.5','UNDER2.5'})
