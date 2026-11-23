# Serie A / Sisal — analisi matematica delle quote

Piccolo progetto di ricerca in Python. Importa il foglio pubblico delle quote **Sisal**, legge solo le partite di Serie A, stima le probabilità con Poisson e genera una tabella con 1X2 e Under/Over 2,5. **Non effettua scommesse** e non promette guadagni.

## Utilizzo sul Mac (da Terminale)

Estrai lo ZIP, apri Terminale dentro la cartella ed esegui:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_seriea.txt
python seriea_value_model.py sisal --bankroll 60
```

Il comando `sisal` tenta automaticamente di:

1. Scaricare il PDF ufficiale **Calcio Base per Manifestazione** da Sisal.
2. Estrarre calendario, squadre, 1X2, Under/Over 2,5 dalla sezione *ITA Serie A*.
3. Verificare la data di aggiornamento stampata in fondo al PDF; **se è più vecchio di 48 ore, interrompere l'analisi**.
4. Scaricare i CSV storici da Football-Data e integrare soltanto risultati mancanti verificati dal feed secondario open-matchday. Se il confronto delle fonti o il calendario non torna, produrre soltanto un report esplorativo, **senza candidate né puntate**.
5. Calcolare probabilità, quote eque, valore atteso (EV), probabilità del mercato senza margine e una eventuale puntata teorica.

**Non devi compilare manualmente il CSV delle partite.**

### File prodotti

In `seriea_data/` trovi:

- `sisal_foglio_quote.pdf`: PDF scaricato da Sisal.
- `sisal_quote_importate.csv`: gli incontri e le quote realmente estratti dal PDF.
- `risultati_integrati_open_matchday.csv`: i soli risultati aggiunti dal feed secondario (vuoto quando non servono).
- `sisal_analisi.csv`: tutti i mercati analizzati (comprese le scelte **NO_BET**). Se lo storico è vecchio e non siamo in una sosta verificata, ogni riga riporta `STORICO_DA_AGGIORNARE` e una puntata teorica di **0 EUR**.
- `sisal_report.md`: il riepilogo leggibile delle sole candidate matematiche.

Il foglio PDF è uno **snapshot**, non un feed live. Le quote possono cambiare prima della partita: verifica sempre la quota effettiva. I contenuti e il formato del sito sono sotto controllo di Sisal: se cambiano, l'importatore potrebbe richiedere modifiche.

### Se il download automatico non funziona

Scarica il PDF dal [foglio quote Sisal](https://www.sisal.it/scommesse-matchpoint/foglio-quote), scegli **Calcio Base per Manifestazione** e salvalo sul Mac. Poi:

```bash
python seriea_value_model.py sisal --pdf ~/Downloads/nome-del-file.pdf --bankroll 60
```

Il PDF deve essere recente. Se appare l'errore **"Foglio Sisal vecchio"**, scaricane una versione aggiornata: non disattivare il controllo per valutare quote reali.

### Se i nomi delle squadre non coincidono

L'importatore riconosce i nomi più comuni e alcune varianti (es. `Hellas Verona` -> `Verona`). In caso di nome sconosciuto **salta la partita** e lo segnala: non abbina squadre per somiglianza incerta.

## Modello e limiti

- Distribuzione di Poisson indipendente per le reti, con forze di attacco/difesa, vantaggio casa, riduzione del peso delle partite vecchie e penalità L2.
- Mercati: 1, X, 2, OVER2.5, UNDER2.5.
- Probabilità implicita del mercato: inverso quota, normalizzato sul mercato completo (margine rimosso in modo proporzionale).
- EV: `probabilità_del_modello × quota_Sisal - 1`. EV stimato è **solo una previsione**, non una garanzia.
- Candidate se EV >= 5%, vantaggio sulla probabilità normalizzata del mercato >= 3 punti percentuali, almeno 8 partite nello storico per squadra. Queste soglie **non sono state validate contro Sisal**.
- Puntata simulata: un quarto di Kelly con limite dell'1% del bankroll per partita e del 5% esposto in totale (per 60 €, massimo **0,60 € a partita** e **3 € complessivi**). Se Sisal richiede una puntata minima superiore, **non arrotondare in alto**.
- Se non ci sono candidate, il risultato corretto è `NESSUNA CANDIDATA`.

Il modello è una **baseline**, non un modello Dixon–Coles completo: non usa xG, infortuni, quote di apertura vs chiusura, titolari probabili o cambi d'allenatore. Non c'è alcuna prova, al momento, di un vantaggio sistematico contro le quote Sisal. Prima di rischiare denaro è indispensabile un backtest su dati cronologici, idealmente con quote storiche *Sisal effettivamente disponibili in quel momento*.

## Altri comandi

```bash
python seriea_value_model.py fetch
python seriea_value_model.py backtest --start 2025-08-01 --bankroll 60
python -m unittest discover -p 'test_*.py' -v
```

Il backtest esistente usa quote di Football-Data (soprattutto Bet365): **non rappresenta una prova contro Sisal**. La selezione di quote storiche di chiusura può introdurre altri limiti di fattibilità temporale.

Dati e riferimenti:
- Foglio ufficiale Sisal: https://www.sisal.it/scommesse-matchpoint/foglio-quote
- Fonte risultati storici: https://www.football-data.co.uk/italym.php
- Feed secondario risultati 2026/27: https://github.com/oladeeayo/open-matchday (uso personale, non commerciale; attribuzione richiesta).
- Dixon e Coles (1997), *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*.

### Stato dei test

I test automatizzati includono un **PDF artificiale** con la stessa disposizione delle colonne visibile nel documento ufficiale. In questo ambiente il download diretto del PDF reale Sisal non era disponibile, quindi la lettura di un PDF reale aggiornato non è stata verificata da qui. Il programma interrompe l'analisi in caso di formato imprevisto, dati scaduti o quote incomplete, anziché inventare previsioni.

### Aggiornamento automatico dei risultati

Il comando `sisal` usa Football-Data come fonte storica principale e consulta
`open-matchday` per integrare eventuali risultati 2026/27 mancanti. Prima di
aggiungerli controlla: squadre riconosciute, risultati non contraddittori,
nessun duplicato, 10 incontri per giornata considerando partite giocate e
future, assenza di incontri già iniziati senza risultato. Non sostituisce
mai i risultati già presenti in Football-Data.

Se il feed secondario non è disponibile o non supera i controlli, il report
rimane **esplorativo**, con puntate a zero. Non si disattiva la protezione.
Il feed è giornaliero e non live: durante una giornata in corso l'analisi può
restare bloccata finché non vengono pubblicati i risultati.

Esegui sempre lo stesso comando:

```bash
python3 seriea_value_model.py sisal --bankroll 60
```

`--no-refresh` è destinato ai test offline: salta l'aggiornamento, dunque
non va usato per analizzare quote attuali.
