"""All Sisal market outcomes stay visible regardless of betting validation."""
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import pandas as pd

from sisal_reporting import rank_full_board, terminal_board, write_readable_report


class TestSisalProbabilityBoard(TestCase):
    def sample(self):
        return pd.DataFrame([
            dict(Data='2026-11-28',Ora='15:00',Partita='Team A - Team B',
                 Mercato=m,QuotaSisal=odd,ProbMercatoNoVig=p,
                 ProbPoissonDC=dc,QualitaStorico='OK',
                 Stato='NON_VALIDATO_SISAL',PuntataTeorica=0.0)
            for m,odd,p,dc in [
                ('1',1.91,.49,.53),('X',3.4,.27,.24),('2',4.0,.24,.23),
                ('OVER2.5',2.1,.47,.42),('UNDER2.5',1.9,.53,.58)]
        ])

    def test_all_five_market_outcomes_are_preserved(self):
        actual=rank_full_board(self.sample())
        self.assertEqual(len(actual),5)
        self.assertEqual(set(actual.Mercato),
                         {'1','X','2','OVER2.5','UNDER2.5'})
        self.assertEqual(list(actual.Mercato),
                         ['1','X','2','UNDER2.5','OVER2.5'])
        self.assertEqual(list(actual.RankMercato),[1,2,3,1,2])
        self.assertEqual(int(actual.PiuProbabileNelMercato.sum()),2)
        self.assertTrue((actual.PuntataTeorica == 0).all())

    def test_all_odds_in_terminal_and_markdown(self):
        board=rank_full_board(self.sample())
        console=terminal_board(board)
        self.assertIn('1 @ 1.91',console)
        self.assertIn('X @ 3.40',console)
        self.assertIn('2 @ 4.00',console)
        self.assertIn('UNDER2.5 @ 1.90',console)
        self.assertIn('OVER2.5 @ 2.10',console)
        self.assertIn('senza margine',console.lower())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.md'
            write_readable_report(board,path,updated=datetime(2026,11,23,10),
                                  now=datetime(2026,11,23,10,30),
                                  source_url='https://example.org/quotes',
                                  last_result=pd.Timestamp('2026-11-22'),
                                  freshness_reason='Dati verificati',
                                  history_stale=False,unrecognized=[])
            doc=path.read_text()
            self.assertIn('Tutte le quote',doc)
            self.assertIn('Migliore 1X2',doc)
            self.assertIn('UNDER2.5',doc)
            self.assertEqual(doc.count('| 2026-11-28 15:00 | Team A - Team B'),6)

    def test_missing_dc_prediction_does_not_hide_sisal_odds(self):
        sample=self.sample()
        sample['ProbPoissonDC']=float('nan')
        sample['Stato']='SQUADRE_NON_RICONOSCIUTE'
        board=rank_full_board(sample)
        self.assertEqual(len(board),5)
        self.assertIn('DC n/d', terminal_board(board))

    def test_no_cross_market_best_of_five_label(self):
        board=rank_full_board(self.sample())
        top=board[board.PiuProbabileNelMercato]
        self.assertEqual(set(top.GruppoMercato), {'1X2','Under/Over 2.5'})
        self.assertEqual(len(top),2)
