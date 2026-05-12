# DATA_INVENTORY.md

Inventario completo dei dati disponibili nel repo di Shehzad per i tre dataset che
manteniamo in tesi: **Gowalla**, **AmazonBook**, **Tmall**.

Documento prodotto in Fase 1 ispezionando direttamente i file `.pkl` con lo script
[`inspect_pkls.py`](inspect_pkls.py) (vedi anche il loader originale
[`topn_baselines_neurals/Data_manager/Gowalla_AmazonBook_Tmall_DCCF.py`](topn_baselines_neurals/Data_manager/Gowalla_AmazonBook_Tmall_DCCF.py)).

> **Sintesi in una riga.** I `.pkl` di Shehzad contengono **soltanto matrici sparse
> di interazioni binarie utente×item**. Nessun timestamp, nessuna location, nessuna
> categoria, nessuna sessione, nessuna feature di utente o item. Per costruire un
> modello situation-aware sui *suoi* split servono raw dataset esterni e una
> procedura di mapping (vedi sezione finale).

---

## 1. Formato dei file `.pkl`

Tutti e tre i dataset condividono la stessa convenzione, dentro `data/DCCF/<dataset>/`:

| File              | Tipo Python                       | Contenuto                                                                                                |
| ----------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `train.pkl`       | `scipy.sparse.coo_matrix`         | URM di training (utenti × item), `dtype=float64`, **tutti i valori = 1.0** (implicit feedback)           |
| `test.pkl`        | `scipy.sparse.coo_matrix`         | URM di test, stessa forma di `train.pkl`, valori = 1.0                                                   |
| `valid.pkl`       | `scipy.sparse.coo_matrix`         | URM di validazione (**solo AmazonBook**), valori = 1.0                                                   |
| `train_index.pkl` | `numpy.ndarray` shape `(2, nnz)`  | Stack `[rows; cols]` del training in `int32`. È ridondante rispetto a `train.pkl.row` e `train.pkl.col`. |
| `test_index.pkl`  | `numpy.ndarray` shape `(2, nnz)`  | Stesso per il test                                                                                       |
| `valid_index.pkl` | `numpy.ndarray` shape `(2, nnz)`  | Stesso per il valid (solo AmazonBook)                                                                    |

I file `*_index.pkl` non sono mai letti dal codice di Shehzad (`grep` su tutta la code-base
non trova un solo accesso a quei file): sono materiale ridondante.

**Cosa non c'è in alcun `.pkl`:**
- Timestamp delle interazioni
- Location / coordinate / POI
- Categoria, brand, prezzo, descrizione, immagini di item
- Età, sesso, demografica utente
- Identificativo di sessione
- Rating numerici (i valori sono tutti 1.0; il task è implicit-only)
- Mapping verso ID originali del raw dataset (vedi sezione 5)

---

## 2. Gowalla — `data/DCCF/gowalla/`

### File presenti

```
test.pkl        1.99 MB  scipy.sparse.coo_matrix, shape (50821, 57440), nnz=130270
test_index.pkl  0.99 MB  numpy.ndarray (2, 130270) int32
train.pkl      17.89 MB  scipy.sparse.coo_matrix, shape (50821, 57440), nnz=1172425
train_index.pkl 8.95 MB  numpy.ndarray (2, 1172425) int32
```

**Nessun `valid.pkl`.** La validazione viene costruita on-the-fly con un sub-split
deterministico per utente del training (vedi `CODE_ANALYSIS.md`).

### Statistiche misurate (dai `.pkl` direttamente)

| Quantità                       | Train     | Test     | Train + Test |
| ------------------------------ | --------- | -------- | ------------ |
| Utenti totali (shape)          | 50.821    | 50.821   | 50.821       |
| Item totali (shape)            | 57.440    | 57.440   | 57.440       |
| Interazioni                    | 1.172.425 | 130.270  | 1.302.695    |
| Utenti con 0 interazioni       | 0         | 12.587   | —            |
| Interazioni/utente (min/max/μ) | 1/947/23.07 | 0/95/2.56 | —          |
| Interazioni/item (min/max/μ)   | 1/2273/20.41 | 0/240/2.27 | —         |
| `dtype` valori                 | float64 (sempre 1.0) | float64 (sempre 1.0) | — |

**Discrepanze con la tabella ufficiale di Shehzad** ([docs/tables_window/tables_window_DCCF.html](docs/tables_window/tables_window_DCCF.html)
sezione Gowalla, rows ~362-372):
- Item totali: doc → **41.390**, file → **57.440**. La doc ha copia-incollato il valore di
  Tmall. La forma reale del `.pkl` è 50.821 × 57.440.
- `Interactions per user min`: doc → **5**, file → **1**. Probabilmente la doc si
  riferisce a un filtro 5-core applicato in preprocessing che però non è effettivamente
  rispettato nel `.pkl` finale (alcuni utenti hanno una sola interazione di training).
- `Interactions per item min`: doc → **1**, file → **1**. Coerente.
- Testing users: doc → **50.821**, file → **38.234** (= 50.821 - 12.587 con 0 test items).

### Cosa contiene un'interazione

L'esempio della prima riga di `train.pkl.col[:10]` per `row=0`:
```
items = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
data  = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
```
Cioè ogni interazione è la coppia `(user_index, item_index)`, valore 1, e nient'altro.

### File raw o di mapping presenti

Nessuno nella cartella `data/DCCF/gowalla/`. **Non esiste alcun file** che mappi
gli indici 0..50.820 (utenti) e 0..57.439 (item) verso gli ID originali del dataset
SNAP / Brightkite-Gowalla.

> **Nota di contesto:** la cartella `data/DGCF/gowalla/` (che andremo a rimuovere
> in fase di pulizia) contiene `user_list.txt`/`item_list.txt` con un mapping
> `org_id remap_id` verso ID originali, **ma** sono per una versione di Gowalla
> diversa: 29.858 utenti × 40.981 item, processata per DGCF. Non sono direttamente
> riutilizzabili per il DCCF Gowalla (utenti/item diversi).

---

## 3. AmazonBook — `data/DCCF/amazonBook/`

### File presenti

```
test.pkl         9.77 MB  scipy.sparse.coo_matrix, shape (78578, 77801), nnz=640045
test_index.pkl   4.88 MB  numpy.ndarray (2, 640045) int32
train.pkl       34.18 MB  scipy.sparse.coo_matrix, shape (78578, 77801), nnz=2240156
train_index.pkl 17.09 MB  numpy.ndarray (2, 2240156) int32
valid.pkl        4.88 MB  scipy.sparse.coo_matrix, shape (78578, 77801), nnz=320023
valid_index.pkl  2.44 MB  numpy.ndarray (2, 320023) int32
```

**È l'unico dei tre dataset con un `valid.pkl` separato e pre-calcolato.**
Curiosità: nonostante questo, lo script di tuning
[`run_hyperparameter_search_baseline_DCCF_for_datasets.py:38`](run_hyperparameter_search_baseline_DCCF_for_datasets.py:38)
costruisce il validation set con **lo stesso sub-split per utente** usato per Gowalla
e Tmall (vedi `CODE_ANALYSIS.md` §5), ignorando completamente il `valid.pkl` fornito.
Il file è quindi presente nel repo ma **non usato** dalla pipeline di Shehzad.

### Statistiche misurate

| Quantità                       | Train     | Valid    | Test     |
| ------------------------------ | --------- | -------- | -------- |
| Shape                          | 78.578 × 77.801 | 78.578 × 77.801 | 78.578 × 77.801 |
| Interazioni                    | 2.240.156 | 320.023  | 640.045  |
| Utenti con 0 interazioni       | 0         | 8.247    | 1.319    |
| Interazioni/utente (min/max/μ) | 2/7554/28.51 | 0/1091/4.07 | 0/2163/8.15 |
| Interazioni/item (min/max/μ)   | 1/1613/28.79 | 0/214/4.11 | 0/462/8.23 |

Numeri perfettamente allineati con la doc table di Shehzad
(78.578 utenti, 77.801 item, 2.240.156 interazioni di train).

### File raw / mapping

Nessuno. Stesso scenario di Gowalla: nessun mapping verso gli ASIN originali di Amazon.

---

## 4. Tmall — `data/DCCF/tmall/`

### File presenti

```
test.pkl         4.00 MB  scipy.sparse.coo_matrix, shape (47939, 41390), nnz=261939
test_index.pkl   2.00 MB  numpy.ndarray (2, 261939) int32
train.pkl       35.97 MB  scipy.sparse.coo_matrix, shape (47939, 41390), nnz=2357450
train_index.pkl 17.99 MB  numpy.ndarray (2, 2357450) int32
```

**Nessun `valid.pkl`**, come Gowalla.

### Statistiche misurate

| Quantità                       | Train     | Test     |
| ------------------------------ | --------- | -------- |
| Shape                          | 47.939 × 41.390 | 47.939 × 41.390 |
| Interazioni                    | 2.357.450 | 261.939  |
| Utenti con 0 interazioni       | 0         | 411      |
| Interazioni/utente (min/max/μ) | 14/232/49.18 | 0/32/5.46 |
| Interazioni/item (min/max/μ)   | 3/2173/56.96 | 0/251/6.33 |

Numeri coerenti con la doc table di Shehzad (47.939 utenti, 41.390 item,
2.357.450 interazioni).

### File raw / mapping

Nessuno. Tmall ha la peculiarità di essere — nella forma raw IJCAI'15 — un dataset
ricco di feature (categoria, brand, action_type=click/cart/fav/buy, timestamp).
**Tutta questa informazione è stata appiattita** nella versione `.pkl` di Shehzad,
che conserva solo coppie utente-item con valore 1.

---

## 5. Feature contestuali e metadata: cosa abbiamo davvero

**TL;DR.** Per modellare situazioni nel senso classico (contesto temporale, location,
attività, ecc.) sui dataset di Shehzad: **non c'è nulla nei `.pkl`**. È il piano B.

| Segnale contestuale che potrebbe servirci      | Presente in `.pkl`? | Presente in raw originale? | Recuperabile?                            |
| ---------------------------------------------- | ------------------- | -------------------------- | ---------------------------------------- |
| Timestamp dell'interazione                     | ❌ No               | Sì (tutti e tre)           | ⚠️ Solo con re-mapping ID originali     |
| Location (lat/lon) — solo Gowalla              | ❌ No               | Sì (Gowalla raw)           | ⚠️ Solo con re-mapping ID originali     |
| Categoria item                                 | ❌ No               | Sì (AmazonBook, Tmall)     | ⚠️ Solo con re-mapping ID originali     |
| Brand item                                     | ❌ No               | Sì (Tmall, AmazonBook)     | ⚠️ Solo con re-mapping ID originali     |
| Action type (click/cart/fav/buy) — solo Tmall  | ❌ No               | Sì (Tmall raw)             | ⚠️ Solo con re-mapping ID originali     |
| Demografica utente                             | ❌ No               | Parzialmente (Tmall age)   | ⚠️ Solo con re-mapping ID originali     |
| ID utente / item originali                     | ❌ No               | n/a                        | ❌ **Non recuperabile** dai soli `.pkl` |

Il blocco è la riga in fondo: **gli indici 0..N-1 nei `.pkl` sono internal index
arbitrari**, e Shehzad non fornisce alcun file che li mappi agli ID originali
SNAP-Gowalla, ASIN-Amazon, IJCAI15-Tmall.

### Conseguenze pratiche per il modello FM situazionale

Tre opzioni possibili, in ordine di realismo crescente:

1. **Plan A — Stare sui `.pkl` di Shehzad usandoli implicit-only.** Comparabile
   numericamente, ma il modello FM si riduce a un FM su `(user, item)` senza
   contesto, perdendo l'intero argomento di tesi.

2. **Plan B — Rilanciare il preprocessing di Shehzad da zero partendo dai raw**
   tenendo i campi contestuali. Lo script di preprocessing non è nel repo: andrebbe
   ricostruito leggendo il paper SIGIR 2025 e confrontando con i raw delle fonti
   citate dai paper originali (DCCF dataset README su GitHub). C'è un rischio di
   non-riproducibilità: nostro preprocessing potrebbe non coincidere bit-perfect
   con quello di Shehzad, quindi le baseline non-neurali tunate non sarebbero più
   direttamente confrontabili con le sue tabelle pubblicate.

3. **Plan C — Plan B parziale + ri-tuning.** Ricostruiamo il dataset con contesto,
   ri-tuniamo le baseline non-neurali sul *nostro* split, riportiamo onestamente
   in tesi le nuove cifre baseline e citiamo quelle di Shehzad come riferimento
   storico. Più onesto metodologicamente, costo computazionale superiore.

> Questa scelta è fuori scope per la Fase 1. Va decisa con Luca prima della
> Fase 3 (design FM situazionale). La Fase 2 (protocollo statistico) può
> procedere indipendentemente su Plan A.

### Fonti raw da cui ripartire (per Plan B/C)

- **Gowalla**: SNAP <https://snap.stanford.edu/data/loc-gowalla.html> — check-in
  con `(user, time, lat, lon, location_id)`. Il paper DCCF originale (HuangChao Hua,
  SIGIR 2023) usa la stessa fonte, applicando 5-core su utenti e item.
- **AmazonBook**: Amazon Reviews <https://nijianmo.github.io/amazon/> categoria
  "Books". Ha `reviewerID, asin, overall, unixReviewTime`. Il paper DCCF originale
  riferisce alla versione "Amazon Reviews 2018" categoria Books con filtri 20-core.
- **Tmall**: IJCAI-15 contest <https://tianchi.aliyun.com/dataset/42> con file
  `user_log_format1.csv` che contiene `(user_id, item_id, cat_id, seller_id,
  brand_id, time_stamp, action_type)`. Il paper DCCF originale specifica il subset
  che usa.

---

## 6. Note operative per chi rilegge

- Il [script di ispezione](inspect_pkls.py) può essere rilanciato in qualsiasi
  momento dalla root del repo (`python3 inspect_pkls.py`). Non modifica nulla.
- I numeri di questa pagina sono presi da una run del 2026-05-12 su MacBook Air
  M2 con Python 3.11 + numpy 1.26.4 + scipy 1.12.0. Sono deterministici: dipendono
  solo dai `.pkl`, che sono frozen.
- I `.pkl` di Shehzad sono caricabili senza dipendenze deep (basta `numpy` + `scipy`)
  perché sono solo sparse matrix. Questo è utile sapere: la Fase 1 e Fase 2 in CPU
  non hanno bisogno dell'env GPU.
