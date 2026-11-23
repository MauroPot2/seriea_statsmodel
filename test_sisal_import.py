"""Tests with a synthetic PDF matching the visible official Sisal table geometry."""
from datetime import date, datetime
import unittest
import fitz
from sisal_import import parse_sisal_pdf, check_freshness, SisalImportError, infer_match_date


def fake_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=842,height=595)
    w=page.rect.width
    def put(x,y,txt,size=7.0):
        page.insert_text((x*w,y),txt,fontsize=size,fontname='helv')
    put(.03,75,'ITA Serie A',8.5)
    rows=[
      ('29/11','15.00','Sassuolo','Napoli','4,50','4,25','1,62','1,57','2,25'),
      ('30/11','20.45','Cagliari','Milan','4,50','3,75','1,72','1,75','1,95'),
    ]
    for y,row in zip((95,109),rows):
        for offset,value in zip((.025,.065,.176,.300,.440,.468,.494,.729,.754),row):
            put(offset,y,value,size=6.5)
    put(.03,145,'ITA Serie B',8.5)
    row=('01/12','20.00','Catanzaro','Modena','2,20','3,10','3,20','1,60','2,10')
    for offset,value in zip((.025,.065,.176,.300,.440,.468,.494,.729,.754),row):
        put(offset,164,value,size=6.5)
    page.insert_text((475,575),'Dati aggiornati al 22/11/2026 - ore 12:00:00',fontsize=6)
    raw=doc.tobytes()
    doc.close()
    return raw

class TestSisalImporter(unittest.TestCase):
    def test_extracts_only_serie_a_and_exact_columns(self):
        fixture,updated=parse_sisal_pdf(fake_pdf(),reference_date=date(2026,11,22))
        self.assertEqual(updated,datetime(2026,11,22,12,0))
        self.assertEqual(len(fixture),2)
        self.assertEqual(fixture[0]['HomeTeamSisal'],'Sassuolo')
        self.assertEqual(fixture[0]['AwayTeamSisal'],'Napoli')
        self.assertAlmostEqual(fixture[0]['OddsH'],4.50)
        self.assertAlmostEqual(fixture[0]['OddsD'],4.25)
        self.assertAlmostEqual(fixture[0]['OddsA'],1.62)
        self.assertAlmostEqual(fixture[0]['OddsUnder25'],1.57)
        self.assertAlmostEqual(fixture[0]['OddsOver25'],2.25)
        self.assertEqual(fixture[0]['Date'],'2026-11-29')
        self.assertEqual(fixture[0]['Time'],'15:00')

    def test_new_year_inference(self):
        self.assertEqual(infer_match_date('3/1',date(2026,12,31)),date(2027,1,3))

    def test_stale_pdf_fails_closed(self):
        with self.assertRaises(SisalImportError):
            check_freshness(datetime(2026,9,9,23,0),datetime(2026,11,22,12,0))

    def test_invalid_pdf_fails(self):
        with self.assertRaises(SisalImportError):
            parse_sisal_pdf(b'html page')

if __name__=='__main__':unittest.main()
