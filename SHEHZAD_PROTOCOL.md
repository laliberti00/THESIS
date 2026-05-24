# SHEHZAD_PROTOCOL.md

Ispezione del protocollo sperimentale di Shehzad, Ferrari Dacrema & Jannach
(SIGIR 2025) per capire **esattamente** cosa fa il loro codice di valutazione,
prima di scrivere il nostro script di validazione statistica.

> **Punto chiave in due righe.** Shehzad **non fa alcun test di significatività
> statistica** e **non salva metriche per-utente**. Riporta un solo numero
> aggregato per coppia (modello, dataset). Il nostro script di Fase 2 aggiungerà
> rigore: paired test, IC bootstrap, multipli seed sui modelli deep.

Tutte le citazioni sono `path/file.py:NN` e fanno riferimento a questa repo.

---

## Q1 — Seed e ripetizioni

**Risposta sintetica.** Una sola run per modello. Seed singolo, hardcoded
nel parser. Nessun loop su seed multipli, nessuna media tra ripetizioni.

### Dove sono i seed e quali valori usano

| Modello                       | Seed         | Dove è hardcoded                                                                          | Note                                                                                  |
| ----------------------------- | ------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **DCCF**                      | `2022`       | [`topn_baselines_neurals/Recommenders/DCCF/utility/parser.py:6`](topn_baselines_neurals/Recommenders/DCCF/utility/parser.py:6) | `parser.add_argument('--seed', type=int, default=2022)`. Applicato a `random`, `numpy`, `torch`, `torch.cuda` in `DCCF/DCCF_main.py:14-17`. |
| **BIGCF**                     | `2023`       | [`topn_baselines_neurals/Recommenders/BIGCF/utility/parser.py:6`](topn_baselines_neurals/Recommenders/BIGCF/utility/parser.py:6) | Stesso pattern in `BIGCF/BIGCF_main.py:16-21`.                                        |
| **Random** baseline           | `42`         | [`topn_baselines_neurals/Recommenders/NonPersonalizedRecommender.py:161`](topn_baselines_neurals/Recommenders/NonPersonalizedRecommender.py:161) | `def fit(self, random_seed=42): np.random.seed(random_seed)`.                         |
| TopPop, ItemKNN, UserKNN, P3α, RP3β, EASE^R | nessun seed esplicito | n/a                                                                          | Sono **deterministici** dato lo split. La loro riproducibilità non dipende da seed.   |
| **HP search (skopt)**         | dinamico     | [`SearchBayesianSkopt.py:95`](topn_baselines_neurals/HyperparameterTuning/SearchBayesianSkopt.py:95) | `self.random_state = int(os.getpid() + time.time()) % np.iinfo(np.int32).max`. **Cambia ad ogni run**. Significa che la HP search non è bit-identica fra esecuzioni successive (ma il risultato finale che leggiamo è hardcoded in `run_experiments_for_DCCF_original_baselines.py:84-100`, quindi per noi è frozen). |

### Numero di ripetizioni

- **Una sola.** Cercando nel repo `seed_list`, `seeds = [`, `for seed in`,
  `range(... seed`, `n_runs`, `num_runs`, `repetitions`, `n_repeat` non si
  trova **nessun** loop su seed multipli per la valutazione finale.
- Gli script `run_experiments_for_*_original*.py` chiamano il training del
  modello una volta sola con il seed di default.

**Conseguenza per noi.** I numeri pubblicati da Shehzad per DCCF/BIGCF sono
un singolo point estimate, senza varianza riportata. Il nostro script di
Fase 2 dovrà:
- per i modelli deep: rilanciare con almeno 5 seed (es. 2022, 2023, 42, 0, 1)
  e calcolare media + std + IC;
- per le baseline deterministe: una sola run è sufficiente (non c'è varianza
  da stimare), ma il paired test sui per-utente resta valido.

---

## Q2 — Metriche e cutoff

**Risposta sintetica.** 7 metriche, 7 cutoff (in eval finale); 1 cutoff in HP search.

### Metriche calcolate da `EvaluatorHoldout`

Definite nell'enum [`EvaluatorMetrics`](topn_baselines_neurals/Evaluation/Evaluator.py:21-51).
Quelle **attive** (non commentate) e quindi effettivamente calcolate:

| Metrica           | Tipo                  | Implementazione                                                                                          |
| ----------------- | --------------------- | -------------------------------------------------------------------------------------------------------- |
| `PRECISION`       | accuracy              | `precision()` in [`metrics.py`](topn_baselines_neurals/Evaluation/metrics.py)                            |
| `RECALL`          | accuracy              | `recall()`                                                                                               |
| `MAP`             | rank-aware            | classe `MAP` in `metrics.py:37-60`                                                                       |
| `MRR`             | rank-aware            | classe `MRR` in `metrics.py:117-142`                                                                     |
| `NDCG`            | rank-aware (graded)   | `ndcg()`                                                                                                 |
| `F1`              | accuracy              | calcolata in `get_result_string_df` come 2·P·R/(P+R)                                                     |
| `NOVELTY`         | beyond-accuracy       | classe `Novelty` (richiede `URM_train` per stimare popolarità)                                           |
| `COVERAGE_ITEM`   | beyond-accuracy       | classe `Coverage_Item`                                                                                   |

Metriche **commentate fuori** (presenti nel codice ma non calcolate):
`PRECISION_RECALL_MIN_DEN`, `HIT_RATE`, `ARHR_ALL_HITS`, `MAP_MIN_DEN`,
`AVERAGE_POPULARITY`, `DIVERSITY_GINI`, `SHANNON_ENTROPY`,
`DIVERSITY_HERFINDAHL`, `DIVERSITY_MEAN_INTER_LIST`, `COVERAGE_ITEM_HIT`,
`COVERAGE_USER`, `COVERAGE_USER_HIT`, `ITEMS_IN_GT`, `USERS_IN_GT`,
`DIVERSITY_SIMILARITY`, `RATIO_*` (5 varianti).
Vedi [`Evaluator.py:22-51` e `:357-386`](topn_baselines_neurals/Evaluation/Evaluator.py).

### Cutoff (top-K)

| Fase                            | Cutoff                                  | Dove                                                                                                                        |
| ------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Eval finale DCCF + baseline     | `[1, 5, 10, 20, 40, 50, 100]`          | [`run_experiments_for_DCCF_original_baselines.py:29`](run_experiments_for_DCCF_original_baselines.py:29) (`--Ks` default)   |
| Eval finale BIGCF               | `[1, 5, 10, 20, 40, 50, 100]`          | [`run_experiments_for_BIGCF_original.py:15`](run_experiments_for_BIGCF_original.py:15)                                       |
| **HP search**                   | `[20]` soltanto                         | [`run_hyperparameter_search_baseline_DCCF_for_datasets.py:67`](run_hyperparameter_search_baseline_DCCF_for_datasets.py:67)  |

### Metrica ottimizzata in HP search

`metric_to_optimize = "RECALL"`, `cutoff_to_optimize = 20`. Vedi
[`run_hyperparameter_search_baseline_DCCF_for_datasets.py:68-69`](run_hyperparameter_search_baseline_DCCF_for_datasets.py:68).
Cioè la Bayesian search massimizza **solo Recall@20** sul sub-split di
validazione. Gli altri 6 K e le altre metriche non guidano la scelta degli HP.

---

## Q3 — Test di significatività statistica ⚠️ **PUNTO CRITICO**

**Risposta sintetica. ZERO test di significatività in tutto il repo.**

### Evidenza

Ricerca esaustiva su pattern noti di test inferenziali:

```bash
grep -rn --include="*.py" -E \
  "ttest|wilcoxon|mannwhitneyu|kruskal|friedman|p_value|pvalue|\
   confidence.interval|scipy\.stats|bootstrap|significance" .
```

**Tutti i match sono falsi positivi:**
- `from unittest` / `class MyTestCase(unittest.TestCase)` / `unittest.main()`
  in [`DataIO.py:280-346`](topn_baselines_neurals/Recommenders/DataIO.py:280),
  [`Compute_similarity_test.py`](topn_baselines_neurals/Recommenders/Similarity/Compute_similarity_test.py),
  [`metrics.py:1085-1189`](topn_baselines_neurals/Evaluation/metrics.py:1085)
  → sono test unitari del codice, non test statistici.
- `"Test \t\tquota"`, `"Test \t\tinteractions"` in
  [`DataSplitter_Holdout.py:110`](topn_baselines_neurals/Data_manager/DataSplitter_Holdout.py:110)
  → sono **stringhe di log** che dicono "Test set: …".
- `sys.stderr.flush()` → cattura del pattern `std` o `sem` ma è I/O, non statistica.

**Zero occorrenze di:**
- `from scipy.stats import ...`
- `import scipy.stats`
- Funzioni come `ttest_rel`, `ttest_ind`, `wilcoxon`, `mannwhitneyu`,
  `friedmanchisquare`, `bootstrap`, `permutation_test`.
- Calcolo di varianza/IC sui risultati (`np.std`, `np.var`, `sem`, `np.percentile`
  applicati alle metriche).

### Cosa Shehzad pubblica

Solo **medie aggregate**: un numero per `(modello, dataset, cutoff, metrica)`.
Le tabelle in [`docs/tables_window/tables_window_DCCF.html`](docs/tables_window/tables_window_DCCF.html)
e nei file `results/DCCF/*.txt` riportano valori grezzi senza barre d'errore,
senza p-value, senza indicazione di significatività.

### Conseguenza per la Fase 2

**Il nostro script di validazione statistica aggiungerà rigore rispetto al
protocollo originale**, non lo replicherà soltanto. Va dichiarato esplicitamente
in tesi che:

- noi *aggiungiamo* il protocollo statistico assente in Shehzad;
- la mancanza di test in Shehzad è un limite metodologico del paper originale,
  non un'omissione voluta da noi.

Il nostro script dovrà:
1. raccogliere metriche **per-utente** (vedi Q4) — Shehzad non lo fa, quindi
   dovremo intervenire sul codice;
2. eseguire test paired (es. Wilcoxon signed-rank, paired t-test) tra coppie
   (modello A, modello B) sul vettore di metriche per-utente;
3. per i modelli stocastici (DCCF, BIGCF) raccogliere risultati su più seed e
   calcolare IC bootstrap o IC parametrici.

---

## Q4 — Metriche per-utente ⚠️ **PUNTO CRITICO**

**Risposta sintetica. NO, le metriche per-utente non vengono mai salvate.
Vengono accumulate come somma scalare e poi divise per `n_users`. I valori
individuali sono *scartati* durante la run.**

### Evidenza nel codice

#### Per Precision, Recall, NDCG: accumulo via `+=`

[`Evaluator.py:357-360`](topn_baselines_neurals/Evaluation/Evaluator.py:357):

```python
results_current_cutoff[EvaluatorMetrics.PRECISION.value] += precision(...)
results_current_cutoff[EvaluatorMetrics.RECALL.value]    += recall(...)
results_current_cutoff[EvaluatorMetrics.NDCG.value]      += ndcg(...)
```

Sono `+=` su uno **scalare**. Nessuna struttura dati salva il valore per
utente.

Poi in [`Evaluator.py:282`](topn_baselines_neurals/Evaluation/Evaluator.py:282):

```python
results_current_cutoff[key] = value / self._n_users_evaluated
```

Cioè la divisione per il numero di utenti viene applicata alla fine. La media
è esposta, la lista per-utente è persa.

#### Per MAP, MRR (e HIT_RATE, anche se non attivo): cumulative + n_users

[`metrics.py:37-60`](topn_baselines_neurals/Evaluation/metrics.py:37) (MAP):

```python
class MAP(_Metrics_Object):
    def __init__(self):
        self.cumulative_AP = 0.0      # solo somma scalare
        self.n_users = 0              # solo conteggio
    def add_recommendations(self, is_relevant, pos_items):
        self.cumulative_AP += average_precision(is_relevant)
        self.n_users += 1
    def get_metric_value(self):
        return self.cumulative_AP / self.n_users
```

Idem `MRR` ([`metrics.py:117-142`](topn_baselines_neurals/Evaluation/metrics.py:117)),
`MAP_MIN_DEN`, `HIT_RATE`. **Nessuna lista per utente** in nessuna di queste
classi.

### Cosa serve per i paired test

I paired test (Wilcoxon signed-rank, paired t-test) richiedono, per ogni
coppia di modelli A e B sullo stesso dataset, **due vettori di lunghezza
`n_users`** con la metrica per utente. Quindi non possiamo applicare i test
sui risultati di Shehzad così come sono — servono modifiche al codice.

### Strategia per Fase 2

Due opzioni tecnicamente equivalenti:

1. **Patchare `EvaluatorHoldout`** aggiungendo accumulatori per-utente
   accanto a quelli aggregati. Modifica intrusiva ma minima (~20 righe in
   `_compute_metrics_on_recommendation_list`). Vantaggio: riusiamo tutta
   l'infrastruttura.
2. **Scrivere un evaluator parallelo** che salva metriche per-utente,
   indipendente da quello di Shehzad. Tipo `PerUserEvaluator`. Vantaggio:
   non tocchiamo il framework upstream; svantaggio: duplichiamo logica.

Da decidere nella Fase 2 quando scriveremo lo script. La mia preferenza è
**opzione 1** con un commit chirurgico, perché preserviamo la coerenza
numerica con Shehzad (stessi valori aggregati, in più i vettori per-utente).

---

## Q5 — Sub-split di validazione

**Risposta sintetica. Quando manca `valid.pkl` (Gowalla, Tmall) — *e anche
quando esiste, per AmazonBook* — la validazione viene costruita on-the-fly
prendendo per ciascun utente l'ultimo 10 % degli item nel suo profilo di
training. Deterministico, non casuale. Il `valid.pkl` di AmazonBook è
ignorato dal codice.**

### Parametri esatti

[`run_hyperparameter_search_baseline_DCCF_for_datasets.py:36-37`](run_hyperparameter_search_baseline_DCCF_for_datasets.py:36):

```python
validation_set    = True
validation_portion = 0.1     # = 10 %
```

[`Gowalla_AmazonBook_Tmall_DCCF.py:27`](topn_baselines_neurals/Data_manager/Gowalla_AmazonBook_Tmall_DCCF.py:27):

```python
def _load_data_from_give_files(self, datapath, validation = False, validation_portion = 0.1):
    # apre solo train.pkl e test.pkl
    # ⚠️ NON apre valid.pkl, anche se per AmazonBook esiste
```

Il valid sub-split è interamente costruito dentro
[`split_train_test_validation` in `DCCF_given_train_test_splits.py:76-103`](topn_baselines_neurals/Data_manager/split_functions/DCCF_given_train_test_splits.py:76).

### Meccanica precisa

Dopo aver costruito `URM_train` (= `URM_all − URM_test`), per ciascun
`user_id` da 0 a `n_users − 1`:

1. Si recuperano gli item del suo profilo di training:
   `user_profile = URM_train.indices[start:end]`.
2. Si calcola `k_out = len(user_profile) − int(len(user_profile) * 0.1)`.
   Quindi `k_out` = "tutto tranne l'ultimo 10 %", parte intera.
3. I primi `k_out` item → `URM_validation_train`.
4. Gli ultimi `len − k_out` (≈ 10 %) → `URM_validation_test`.

### Determinismo

Sì, completamente deterministico. **Non c'è alcuna chiamata a `np.random` o
`random.shuffle`** in questa funzione. L'ordine usato è
`URM_train.indices[start:end]`, cioè l'ordine in cui scipy memorizza i column
index dopo la conversione a CSR: **item_id crescente**, *non* cronologico.

### Implicazione per il modello FM situazionale

Il "10 % finale" mandato in validazione sono gli item con indice più alto
nel `URM_train`. Siccome gli indici Shehzad-internal sono assegnati in modo
non documentato (probabilmente per ordine alfabetico/numerico dell'ID
originale), questa è **una split arbitraria, non temporale**. Se nel modello
FM volessimo introdurre time-aware evaluation, dovremmo costruirci la nostra
split temporale dai raw (Plan B/C), non quella di Shehzad.

---

## Q6 — Hyperparameter search

**Risposta sintetica.** Bayesian search via `scikit-optimize` (`skopt.gp_minimize`),
100 trial × 5 modelli, ottimizza Recall@20 sul sub-split di validazione,
parallelizzato per modello via `multiprocessing.Pool`.

### Libreria e parametri

| Item                  | Valore                                                                                                                          | Dove                                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Algoritmo             | `gp_minimize` (Gaussian Process)                                                                                                | [`SearchBayesianSkopt.py:9`](topn_baselines_neurals/HyperparameterTuning/SearchBayesianSkopt.py:9) (`from skopt import gp_minimize`) |
| `n_cases` (n_calls)   | **100**                                                                                                                         | [`run_hyperparameter_search_baseline_DCCF_for_datasets.py:71`](run_hyperparameter_search_baseline_DCCF_for_datasets.py:71) |
| `n_random_starts`     | **5**                                                                                                                           | stessa riga 72                                                                                                              |
| `acq_func`            | `gp_hedge` (default skopt)                                                                                                      | [`SearchBayesianSkopt.py:76`](topn_baselines_neurals/HyperparameterTuning/SearchBayesianSkopt.py:76)                       |
| `xi`                  | 0.01 (exploitation/exploration trade-off)                                                                                       | stesso file riga 80                                                                                                          |
| `kappa`               | 1.96                                                                                                                            | stesso file riga 81                                                                                                          |
| `noise`               | `1e-5`                                                                                                                          | stesso file riga 75                                                                                                          |
| `random_state`        | `int(os.getpid() + time.time()) % np.iinfo(np.int32).max` — **dinamico, cambia ogni run**                                       | stesso file riga 95                                                                                                          |
| Parallelizzazione     | `multiprocessing.Pool(processes=cpu_count())` — un processo per modello                                                         | [`run_hyperparameter_search_baseline_DCCF_for_datasets.py:93`](run_hyperparameter_search_baseline_DCCF_for_datasets.py:93) |

### Spazio di ricerca

#### KNN (`ItemKNN`, `UserKNN`)

[`run_hyperparameter_search.py:53-57`](topn_baselines_neurals/HyperparameterTuning/run_hyperparameter_search.py:53):

```python
"topK":      Integer(5, 1000)
"shrink":    Integer(0, 1000)
"similarity": Categorical([similarity_type])    # eseguito una volta per ogni: cosine, jaccard, asymmetric, dice, tversky
"normalize":  Categorical([True, False])
```

Per `asymmetric` aggiunto `asymmetric_alpha ∈ Real(0, 2)`; per `tversky`
aggiunti `tversky_alpha`, `tversky_beta ∈ Real(0, 2)`.

Se `allow_weighting=True` (è il default in DCCF), in più:
`feature_weighting ∈ Categorical(['none', 'BM25', 'TF-IDF'])`.

#### P3α ([`run_hyperparameter_search.py:253-257`](topn_baselines_neurals/HyperparameterTuning/run_hyperparameter_search.py:253))

```python
"topK":                  Integer(5, 1000)
"alpha":                 Real(0, 2, prior='uniform')
"normalize_similarity":  Categorical([True, False])
```

#### RP3β ([`run_hyperparameter_search.py:272-277`](topn_baselines_neurals/HyperparameterTuning/run_hyperparameter_search.py:272))

```python
"topK":                  Integer(5, 1000)
"alpha":                 Real(0, 2, prior='uniform')
"beta":                  Real(0, 2, prior='uniform')
"normalize_similarity":  Categorical([True, False])
```

#### EASE^R ([`run_hyperparameter_search.py:289-293`](topn_baselines_neurals/HyperparameterTuning/run_hyperparameter_search.py:289))

```python
"topK":             Categorical([None])                     # fissato
"normalize_matrix": Categorical([False])                    # fissato
"l2_norm":          Real(1e0, 1e7, prior='log-uniform')
```

### Metrica e split di ottimizzazione

- **Optimize**: `RECALL` a `cutoff = 20`
  ([`run_hyperparameter_search_baseline_DCCF_for_datasets.py:68-69`](run_hyperparameter_search_baseline_DCCF_for_datasets.py:68)).
- **Train**: `URM_validation_train` (i primi 90 % del training di Shehzad,
  per utente).
- **Validation eval**: `URM_validation_test` (l'ultimo 10 %, per utente).
- **Refit finale** sugli HP migliori: si riallena su `URM_train_last_test = URM_train`
  (full train Shehzad) e si valuta su `URM_test` (test Shehzad). Vedi
  [`run_hyperparameter_search_baseline_DCCF_for_datasets.py:77-90`](run_hyperparameter_search_baseline_DCCF_for_datasets.py:77).

### Stato in questo repo

**La HP search NON è stata rieseguita**. I valori "best" sono già `hardcoded` in
[`run_experiments_for_DCCF_original_baselines.py:84-100`](run_experiments_for_DCCF_original_baselines.py:84):
Shehzad li ha calcolati una volta e li ha incollati nel codice di eval. Il
nostro `repro_check_baseline.py` legge gli stessi valori.

Se vogliamo ri-tunare (es. in Plan C, Fase 3), basta lanciare:
```bash
python run_hyperparameter_search_baseline_DCCF_for_datasets.py --dataset gowalla
```
Costo: ~ ore (5 modelli × 100 trial), CPU-bound.

---

## Q7 — Output e logging

**Risposta sintetica.** TSV per (dataset, modello) con **una riga per cutoff
e una colonna per metrica**, *solo valori aggregati*. Nessun per-utente,
nessuna deviazione standard, nessun p-value.

### Formato dei file

Esempio (`results/DCCF/gowalla_RP3betaRecommender.txt`):

```
cuttOff   PRECISION   RECALL   MAP   MRR   NDCG   F1   NOVELTY   COVERAGE_ITEM   TrainingTime(s)   TestingTimeforRecords(s)   AverageTestingTimeForOneRecord(s)
1         0.1042...   0.0408   ...   ...   ...    ...   ...      ...             135.34            99.05                       0.0019
5         0.0626...   0.1172   ...   ...   ...    ...   ...      ...             0.0               0.0                         0.0
...
100       0.0141...   0.4765   ...   ...   ...    ...   ...      ...             0.0               0.0                         0.0
```

Generato in [`run_experiments_for_DCCF_original_baselines.py:152`](run_experiments_for_DCCF_original_baselines.py:152):

```python
results_run_1.to_csv(saved_results + "/" + args.dataset + "_" +
                      recommender_class.RECOMMENDER_NAME + ".txt",
                      sep = "\t", index = False)
```

Una **stranezza di formato**: `TrainingTime(s)`, `TestingTimeforRecords(s)`,
`AverageTestingTimeForOneRecord(s)` sono valori scalari per la run intera,
ma vengono ripetuti come "valore vero" sulla riga `cuttOff=1` e zero su
tutte le altre righe (vedi
[`run_experiments_for_DCCF_original_baselines.py:145-147`](run_experiments_for_DCCF_original_baselines.py:145)).
Non è un errore di Shehzad — è la convenzione del framework `topn_baselines_neurals`.

### Cosa è contenuto

- ✅ Tutte le metriche di Q2 (PRECISION, RECALL, MAP, MRR, NDCG, F1, NOVELTY, COVERAGE_ITEM).
- ✅ Tutti i cutoff `[1, 5, 10, 20, 40, 50, 100]`.
- ✅ Tempi di training e di testing.
- ❌ **Nessun valore per-utente.** Confermato in Q4.
- ❌ Nessun valore di varianza/std.
- ❌ Nessuna informazione su HP usati (sono ricavabili leggendo lo script,
  non dal TSV).
- ❌ Nessun seed registrato.

### Cosa scrivono gli script

Lo script HP search produce — quando lo lanci — file aggiuntivi in
`results/DCCF/<dataset>/optimization/` con metadata del BayesianSearch
(non presenti nel repo finché non rieseguiamo la search). Si tratta di:
`<Model>_metadata.zip`, `<Model>_best_model.zip`, `<Model>_best_parameters.zip`,
`<Model>_best_result_validation.zip`, `<Model>_best_result_test.zip`. Vedi
[`SearchBayesianSkopt.py`](topn_baselines_neurals/HyperparameterTuning/SearchBayesianSkopt.py)
per i dettagli.

---

## Riepilogo per la Fase 2

Cosa il protocollo di Shehzad **fa**:
- 1 run con seed singolo (2022 per DCCF, 2023 per BIGCF, deterministico per
  le baseline).
- 7 metriche × 7 cutoff per ciascun (dataset, modello).
- Bayesian search 100 trial × 5 baseline, ottimizza Recall@20.
- Sub-split di validazione per-utente al 90/10, deterministico.
- Output TSV solo aggregato.

Cosa il protocollo di Shehzad **non fa** — e che dobbiamo aggiungere noi:
- Nessun test statistico (paired t-test, Wilcoxon, bootstrap CI).
- Nessuna ripetizione su seed multipli per i modelli stocastici.
- Nessuna metrica per-utente salvata → impossibile applicare paired test
  sui suoi output esistenti senza prima patchare l'evaluator.
- Nessun IC riportato sulle metriche.

**Decisione di design per il nostro script di Fase 2** (da discutere):
1. **Modificare l'`EvaluatorHoldout`** in modo additivo (mantiene tutte le
   metriche aggregate esistenti, in più espone `per_user_metrics`) → opzione
   raccomandata.
2. **Definire le coppie da testare**: ogni baseline non-neurale vs DCCF e vs
   BIGCF, e (a Phase 3) FM-situazionale vs tutte le altre. Test per cutoff
   = 20 sulla metrica Recall e NDCG.
3. **Test scelto**: Wilcoxon signed-rank (non-parametrico, paired). Backup
   con paired t-test su NDCG per controllo.
4. **IC**: bootstrap (n=10000) sulla media per-utente, IC al 95 %.
5. **Seed per i modelli deep**: minimo 5 (2022, 2023, 42, 0, 1). Per le
   baseline deterministiche: 1 run.
