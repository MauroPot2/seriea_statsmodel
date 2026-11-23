"""Prospective Sisal snapshots are never backfilled after kickoff."""
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import TestCase
import pandas as pd
from sisal_history import append_forecasts, audit_history


class TestSisalHistory(TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        self.now=datetime(2026,11,22,15)
        self.report=pd.DataFrame([{
            'Data':'2026-11-23','Ora':'20:45','HomeTeam':'Inter','AwayTeam':'Roma',
            'Partita':'Inter - Roma','Mercato':market,
            'QuotaSisal':price,'ProbModello':p,'ProbMercatoNoVig':m,
            'QualitaStorico':'OK','ValidazioneSisal':'ASSENTE',
            'AggiornamentoSisal':'2026-11-22T14:45','PuntataTeorica':0.
        } for market,price,p,m in [
            ('1',2.0,.48,.45),('X',3.5,.28,.28),('2',3.4,.24,.27),
            ('OVER2.5',2.1,.47,.49),('UNDER2.5',1.9,.53,.51)]])

    def test_archiving_is_idempotent(self):
        self.assertEqual(append_forecasts(self.report,self.path,self.now),5)
        self.assertEqual(append_forecasts(self.report,self.path,self.now),0)
        self.assertEqual(len(pd.read_csv(self.path/'sisal_forecast_archive.csv')),5)

    def test_no_post_kickoff_backfill(self):
        self.assertEqual(append_forecasts(self.report,self.path,
                                         datetime(2026,11,23,21)),0)
        self.assertFalse((self.path/'sisal_forecast_archive.csv').exists())

    def test_audit_does_not_claim_roi(self):
        append_forecasts(self.report,self.path,self.now)
        history=pd.DataFrame([{'Date':pd.Timestamp('2026-11-23'),
            'HomeTeam':'Inter','AwayTeam':'Roma','FTHG':1,'FTAG':0}])
        result=audit_history(history,self.path)
        self.assertEqual(result['n_matches'],1)
        self.assertEqual(result['status'],'SAMPLE_INSUFFICIENT')
        self.assertEqual(result['groups']['1X2']['n'],1)
        self.assertEqual(result['groups']['OU25']['n'],1)
        self.assertNotIn('roi',result)

    def test_adding_new_forecast_columns_preserves_earlier_snapshots(self):
        append_forecasts(self.report,self.path,self.now)
        stored_before=pd.read_csv(self.path/'sisal_forecast_archive.csv')
        new=self.report.copy()
        new['AggiornamentoSisal']='2026-11-22T15:00'
        new['ProbXGElo']=0.42
        appended=append_forecasts(new,self.path,datetime(2026,11,22,16))
        self.assertEqual(appended,5)
        after=pd.read_csv(self.path/'sisal_forecast_archive.csv')
        self.assertEqual(len(after),10)
        self.assertTrue(after.iloc[:5].ProbXGElo.isna().all())
        self.assertTrue((after.iloc[5:].ProbXGElo==0.42).all())
        self.assertTrue((stored_before.ProbModello.to_numpy() ==
                         after.iloc[:5].ProbModello.to_numpy()).all())

    def test_empty_archive(self):
        self.assertEqual(audit_history(pd.DataFrame(),self.path)['n_matches'],0)
