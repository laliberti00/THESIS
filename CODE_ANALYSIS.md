# CODE_ANALYSIS.md

Analisi sistematica del framework di Shehzad, Ferrari Dacrema & Jannach,
"A Worrying Reproducibility Study of Intent-Aware Recommendation Models" (SIGIR 2025).
Documento autoesplicativo: per leggerlo non serve aver aperto il codice.

Stato a cui si riferisce questo file: branch `clean-thesis-setup`, **prima** della
pulizia. Vedi commit tag `original-shehzad-v1` per lo stato originale immutato.

---

## 1. Struttura del repo

```
IntentAwareRS_original/
├── README.md                                # README HTML originale di Shehzad
├── intentAware.webp, ACM SIGIR 2025*.{pdf,pptx}   # materiale per la submission
├── requirements_gpu.txt                     # dipendenze del main env (Python 3.8 + torch CUDA)
├── requirements_IntentAwareRS_DGCF.txt      # dipendenze separate per DGCF (Python 3.6 + TF 1.14)
│
├── data/                                    # tutti i dataset, organizzati per paper
│   ├── DCCF/{gowalla,amazonBook,tmall}/     # .pkl di Shehzad (vedi DATA_INVENTORY.md)
│   ├── DGCF/{gowalla,amazonBook,yelp2018}/  # txt + npz precomputed adj matrices
│   ├── ID4SNR/{Beauty,MovieLens,Music}.pkl  # .pkl monolitici
│   └── KGIN/{alibabaFashion,amazonBook,lastFm}/ # train.txt + test.txt + kg_final.txt
│
├── docs/                                    # mini-site con tabelle dei risultati (HTML)
│   ├── index.html
│   ├── tables_window/                       # tabelle per ogni paper, finestre temporali
│   └── tables_single/                       # tabelle singole
│
├── log/                                     # log testuali grezzi delle run di Shehzad
│   └── {gowalla,tmall,amazonbook}.log
│
├── results/                                 # risultati canonici prodotti dagli script run_*
│   ├── DCCF/   <dataset>_<Model>.txt        # metriche per cutoff [1,5,10,20,40,50,100]
│   ├── BIGCF/
│   ├── DGCF/
│   ├── ID4SNR/
│   └── KGIN/
│
├── topn_baselines_neurals/                  # framework Python core (fork di RecSys2019_DL_Evaluation)
│   ├── Data_manager/                        # loader di dataset + utility di splitting
│   │   ├── Gowalla_AmazonBook_Tmall_DCCF.py # loader specifico per i pkl di Shehzad
│   │   ├── DatasetMapperManager.py
│   │   └── split_functions/
│   │       └── DCCF_given_train_test_splits.py  # split + sub-split per validazione (vedi §5)
│   ├── Evaluation/
│   │   ├── Evaluator.py                     # EvaluatorHoldout (vedi §6)
│   │   └── metrics.py                       # implementazioni di Recall, NDCG, MAP, MRR, ecc.
│   ├── HyperparameterTuning/
│   │   ├── SearchBayesianSkopt.py           # wrapper di scikit-optimize
│   │   └── run_hyperparameter_search.py     # entry per HP search (vedi §7)
│   ├── Recommenders/                        # implementazioni dei modelli
│   │   ├── NonPersonalizedRecommender.py    # TopPop, Random, GlobalEffects
│   │   ├── KNN/{Item,User}KNNCFRecommender.py
│   │   ├── GraphBased/{P3alpha,RP3beta}Recommender.py
│   │   ├── EASE_R/EASE_R_Recommender.py
│   │   ├── DCCF/                            # codice originale degli autori DCCF
│   │   ├── BIGCF/                           # codice originale degli autori BIGCF
│   │   ├── DGCF_SIGIR_20/                   # codice originale DGCF (TF 1.14)
│   │   ├── IDS4NR/                          # codice originale IDS4NR
│   │   ├── Knowledge_Graph_based_Intent_Network_KGIN_WWW/
│   │   ├── FactorizationMachines/           # contiene LightFMRecommender (non usato qui)
│   │   └── Similarity/
│   │       ├── Compute_Similarity.py        # dispatcher Python/Cython
│   │       └── Compute_Similarity_Python.py
│   ├── CythonCompiler/                      # build script per le estensioni .pyx
│   ├── Cython_examples/                     # esempi non utilizzati
│   ├── Notebooks_utils/                     # utility da notebook (non in production)
│   └── Utils/
│
└── run_experiments_*.py / run_hyperparameter_search_*.py
                                             # entry-point CLI, uno per paper (vedi §2)
```

---

## 2. Punti di ingresso

Ci sono **due classi** di script CLI nella root:

### Script di esperimento (eval finale con HP best hardcoded)

| Script                                              | Modelli interessati                         | Dataset                                |
| --------------------------------------------------- | ------------------------------------------- | -------------------------------------- |
| `run_experiments_for_DCCF_original_baselines.py`    | DCCF + Random, TopPop, ItemKNN, UserKNN, P3α, RP3β, EASE^R | gowalla / amazonBook / tmall |
| `run_experiments_for_BIGCF_original.py`             | Solo BIGCF (riusa gli split DCCF)            | gowalla / amazonBook / tmall |
| `run_experiments_for_DGCF_original.py`              | Solo DGCF                                    | yelp2018 / gowalla / amazonbook |
| `run_experiments_DGCF_baselines.py`                 | Baseline tunate per DGCF                     | yelp2018 / gowalla / amazonbook |
| `run_experiments_for_KGIN_original_baselines.py`    | KGIN + baseline                              | alibabaFashion / amazonBook / lastFm |
| `run_experiments_IDS4NR_original_baselines.py`      | IDS4NR + baseline                            | MovieLens / Beauty / Music |

### Script di hyperparameter search (Bayesian, una volta sola)

| Script                                                       | Cosa tuna                              |
| ------------------------------------------------------------ | -------------------------------------- |
| `run_hyperparameter_search_baseline_DCCF_for_datasets.py`    | Baseline non-neurali sui dataset DCCF  |
| `run_hyperparameter_search_baseline_models_DGCF.py`          | Baseline non-neurali sui dataset DGCF  |
| `run_hyperparameter_search_baseline_models_IDS4NR.py`        | Baseline non-neurali sui dataset IDS4NR |

> Per la nostra tesi ci interessano **solo i due script DCCF**: il tuning produce
> i best-HP che poi vengono **hardcoded** dentro `run_experiments_for_DCCF_original_baselines.py`
> alle righe 84-100, e l'esperimento finale li rilegge da lì.

### Esempi di comandi (post-pulizia, sulla nostra branch)

```bash
# Tuning baseline (lo facciamo solo se vogliamo ri-tunare; per la repro check non serve)
python run_hyperparameter_search_baseline_DCCF_for_datasets.py --dataset gowalla

# Esperimento finale: DCCF + baseline
python run_experiments_for_DCCF_original_baselines.py --dataset gowalla

# Solo BIGCF
python run_experiments_for_BIGCF_original.py --dataset gowalla
```

> Cosa noi abbiamo aggiunto: lo script ausiliario `repro_check_baseline.py` esegue
> **solo le baseline non-neurali** (omette DCCF, che richiede GPU). È identico per
> sostanza a Shehzad ma più ergonomico per i nostri esperimenti CPU di Fase 1.

---

## 3. Flusso di esecuzione (DCCF + baseline, dataset Gowalla)

Tracciato passo per passo da `run_experiments_for_DCCF_original_baselines.py`:

1. **Parsing argomenti** (`--dataset`, `--Ks`).
2. **Caricamento dataset**:
   - Istanzia `Gowalla_AmazonBook_Tmall_DCCF` (loader).
   - Chiama `._load_data_from_give_files(data_path, validation=False)` →
     ritorna `(URM_train, URM_test)` come `scipy.sparse.csr_matrix`.
3. **Training DCCF**:
   - Prima passa: `model_tuningAndTraining(validation=True, ...)` per stimare il
     `best_epoch` su un sub-split di validazione.
   - Seconda passa: `model_tuningAndTraining(validation=False, epoch=best_epoch)`
     allena su tutto il train e valuta su test.
4. **Training baseline (loop su `recommender_class_list`)**:
   - Per ogni classe: istanzia, chiama `.fit(**hp_best_per_dataset)`, valuta con
     `EvaluatorHoldout(URM_test, cutoff=[1,5,10,20,40,50,100], exclude_seen=True)`.
   - Salva il `pd.DataFrame` dei risultati in `results/DCCF/<dataset>_<RecommenderName>.txt`
     come TSV.
5. **Note d'implementazione importanti**:
   - DCCF e BIGCF richiedono CUDA o ne soffrono molto in tempi senza GPU.
   - I tempi di training/eval delle baseline sono nell'ordine dei minuti su CPU.
   - **EASE^R necessita di una Gram matrix densa `n_items × n_items`**: per i nostri
     dataset (≥ 41k item) significa ≥ 14 GB di RAM, non gira su MacBook M2 16 GB.

---

## 4. Modelli implementati (perimetro tesi: DCCF + BIGCF + baseline)

### 4.1 Non-neurali (file Python puri, no Cython, no GPU)

| Modello   | File                                                                       | Interfaccia `.fit()`                                                                          |
| --------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Random    | `Recommenders/NonPersonalizedRecommender.py` (class `Random`)              | nessun HP                                                                                     |
| TopPop    | `Recommenders/NonPersonalizedRecommender.py` (class `TopPop`)              | nessun HP                                                                                     |
| ItemKNNCF | `Recommenders/KNN/ItemKNNCFRecommender.py`                                 | `topK, shrink=1000, similarity='cosine', normalize=True, feature_weighting='TF-IDF'`          |
| UserKNNCF | `Recommenders/KNN/UserKNNCFRecommender.py`                                 | uguale a ItemKNN                                                                              |
| P3α       | `Recommenders/GraphBased/P3alphaRecommender.py`                            | `topK, alpha=1., min_rating=0, implicit=False, normalize_similarity=False`                    |
| RP3β      | `Recommenders/GraphBased/RP3betaRecommender.py`                            | `topK=100, alpha=1., beta=0.6, min_rating=0, implicit=False, normalize_similarity=True`       |
| EASE^R    | `Recommenders/EASE_R/EASE_R_Recommender.py`                                | `topK=None, l2_norm=1e3, normalize_matrix=False`                                              |

> **Trappola di doc:** il file [`docs/tables_window/tables_window_DCCF.html`](docs/tables_window/tables_window_DCCF.html)
> riporta nella tabella "Hyperparameter ranges" un campo `shrink` per i KNN, ma il
> codice di evaluation passa **solo** `topK` e `similarity`, lasciando `shrink=1000`
> al default. Inoltre il valore `feature_weighting='TF-IDF'` **non compare nei docs**
> ma è il default del codice e viene effettivamente usato. Non ignorare questo dettaglio
> nella sezione Methodology del paper di tesi.

> **Altra trappola:** in `run_experiments_for_DCCF_original_baselines.py:99` il dict
> chiamato `RP3alpha_best_HP` è in realtà il dict di P3α (i due nomi sono confondibili).
> Solo `alpha` viene letto da questo dict per P3α, niente `beta` (P3α non ha beta).

### 4.2 Deep intent-aware

| Modello | File principale                                          | Note                                                                       |
| ------- | -------------------------------------------------------- | -------------------------------------------------------------------------- |
| DCCF    | `Recommenders/DCCF/DCCF_main.py` + `model.py`            | PyTorch. `seed=2022`, `epoch=500`, `lr=0.001`. Hardcoded in `parser.py`.   |
| BIGCF   | `Recommenders/BIGCF/...`                                 | PyTorch.                                                                   |

> Le hyperparam di DCCF/BIGCF sono lasciate ai default originali degli autori; sono
> citate (e in alcuni casi sovrascritte) negli script `run_experiments_for_*_original.py`.

### 4.3 Modelli che rimuoveremo nella pulizia

| Modello | Cartella                                                                  | Motivo della rimozione                  |
| ------- | ------------------------------------------------------------------------- | --------------------------------------- |
| DGCF    | `Recommenders/DGCF_SIGIR_20/`                                             | Fuori scope tesi, richiede Python 3.6 + TF 1.14 |
| KGIN    | `Recommenders/Knowledge_Graph_based_Intent_Network_KGIN_WWW/`             | Fuori scope tesi                        |
| IDS4NR  | `Recommenders/IDS4NR/`                                                    | Fuori scope tesi                        |
| LightFM | `Recommenders/FactorizationMachines/LightFMRecommender.py`                | **Da valutare:** è un FM, potenzialmente utile come baseline per la Fase 3. *Lasciamolo per ora* — vedi PHASE1_SUMMARY.md |

---

## 5. Loader dati e splitting

### 5.1 Dataset → URM

Loader: [`topn_baselines_neurals/Data_manager/Gowalla_AmazonBook_Tmall_DCCF.py`](topn_baselines_neurals/Data_manager/Gowalla_AmazonBook_Tmall_DCCF.py).

Sequenza:
1. Apre `train.pkl` e `test.pkl` come `scipy.sparse.coo_matrix`.
2. Costruisce due dizionari `{user_id: [item_ids...]}` dalle interazioni della COO.
3. Concatena train + test in un singolo `URM_dataframe(UserID, ItemID, Data=1)`.
4. Passa il dataframe a `DatasetMapperManager` che crea un internal index
   `user_original_ID_to_index` e `item_original_ID_to_index` (string → int).
   ⚠️ Gli "original ID" qui sono già gli indici Shehzad-internal (0, 1, 2, ...),
   non quelli del raw dataset (Gowalla/AmazonBook/Tmall originali).
5. Chiama `split_train_test_validation` per ricostruire `URM_train` e `URM_test`
   come `csr_matrix`. Se `validation=True`, ritorna anche `URM_validation_train`,
   `URM_validation_test`.

### 5.2 Split train/test/(valid)

File: [`topn_baselines_neurals/Data_manager/split_functions/DCCF_given_train_test_splits.py`](topn_baselines_neurals/Data_manager/split_functions/DCCF_given_train_test_splits.py).

- `URM_test` = la `test.pkl` di Shehzad, mappata sugli indici interni.
- `URM_train` = `URM_all - URM_test` (cioè il complemento del test nell'unione train∪test).
- `URM_validation_train`, `URM_validation_test`:
  - Per ciascun utente, prende il suo profilo in `URM_train`.
  - `k_out = floor(len * (1 - 0.1))` (default `validation_portion=0.1`).
  - I primi `k_out` item → `URM_validation_train`.
  - Gli ultimi `len - k_out` item (≈ 10%) → `URM_validation_test`.

### 5.3 Validation strategy — ATTENZIONE METODOLOGICA

> **Punto critico per il paper di tesi.** Gowalla e Tmall *non hanno* un `valid.pkl`
> dedicato; AmazonBook ce l'ha ma **non viene mai letto dal codice** (vedi `DATA_INVENTORY.md` §3).
> In tutti e tre i casi la validazione è prodotta on-the-fly tramite il sub-split
> per utente descritto sopra. **Non è una split temporale**: l'ordine in `URM.indices`
> dopo CSR è item_id crescente, non chronological.
>
> Conseguenze:
> - Il 10% di validazione "ultimi item" sono in realtà gli **item con indice più
>   alto**, cioè (in DCCF) item meno popolari (gli ID Shehzad sono presumibilmente
>   assegnati ordinando in qualche modo per popolarità o per ordine alfabetico
>   dell'ID originale — non lo sappiamo con certezza, andrebbe verificato se ci
>   serve).
> - Il sub-split è **deterministico** dato lo split train/test → la HP search di
>   Shehzad è riproducibile.
> - Se vogliamo confrontare un modello FM situazionale dovremo replicare lo stesso
>   sub-split usando il loader di Shehzad, non un nostro split casuale.
> - In particolare, il `valid.pkl` di AmazonBook **non va usato** se vogliamo essere
>   numericamente equivalenti — il codice di Shehzad lo ignora di proposito.

---

## 6. Pipeline di evaluation

File: [`topn_baselines_neurals/Evaluation/Evaluator.py`](topn_baselines_neurals/Evaluation/Evaluator.py).

- Classe usata: `EvaluatorHoldout(URM_test, cutoff_list, exclude_seen=True)`.
- `cutoff_list` per gli esperimenti finali: `[1, 5, 10, 20, 40, 50, 100]`.
- Metriche attive (in `EvaluatorMetrics`): **PRECISION, RECALL, MAP, MRR, NDCG, F1, NOVELTY, COVERAGE_ITEM**. Tutte le altre (HIT_RATE, ARHR, DIVERSITY_*, COVERAGE_USER, RATIO_*) sono commentate fuori.
- `exclude_seen=True` → ai punteggi degli item già nel train viene assegnato `-inf`
  prima del top-K, quindi un item visto in train non può essere raccomandato.
- Per ogni utente, l'evaluator chiede `_compute_item_score(user_id_array)` al recommender,
  applica `exclude_seen`, prende il top-K via `np.argpartition`, e accumula le metriche
  iterativamente. Per cutoff multipli la stessa raccomandazione viene tagliata a
  diversi K senza ricalcolarla.
- **Tie-breaking**: `np.argpartition`/`argsort` ordinano per valore decrescente; gli
  item con stesso score sono ordinati in modo dipendente dall'implementazione di numpy.
  Versioni diverse di numpy (1.23 vs 1.26) producono lievi differenze (~1e-5) nelle
  metriche rank-aware (NDCG, MAP, MRR), ma **non** in Recall/Precision (che dipendono
  solo dall'insieme top-K, non dall'ordinamento interno). Vedi `REPRODUCIBILITY_CHECK.md`.

---

## 7. Hyperparameter search

File principale: [`topn_baselines_neurals/HyperparameterTuning/run_hyperparameter_search.py`](topn_baselines_neurals/HyperparameterTuning/run_hyperparameter_search.py).

- **Libreria**: `scikit-optimize` (`skopt`), Bayesian search.
- **Configurazione**: `n_cases=100, n_random_starts=5`, parallelizzato via `multiprocessing.Pool`.
- **Spazio di ricerca KNN** (in `run_hyperparameter_search.py:run_KNNRecommender_on_similarity_type`):
  - `topK ∈ Integer(5, 1000)`
  - `shrink ∈ Integer(0, 1000)`
  - `similarity ∈ {cosine, jaccard, asymmetric, dice, tversky}` (configurato in
    `run_hyperparameter_search_baseline_DCCF_for_datasets.py:89`)
  - `normalize ∈ Categorical([True, False])`
- **Spazio P3α / RP3β** (dentro `SearchAbstractClass`):
  - `topK ∈ Integer(5, 1000)`
  - `alpha ∈ Real(0, 2)` (uniforme)
  - `beta ∈ Real(0, 2)` (uniforme, solo RP3β)
  - `normalize_similarity ∈ Categorical([True, False])`
- **Spazio EASE^R**:
  - `l2_norm ∈ Real(1, 1e7)` con distribuzione log-uniform
  - `normalize_matrix` non viene tunato (resta default)
- **Metric di ottimizzazione**: `RECALL` al `cutoff_to_optimize = 20`.
- **Set su cui si ottimizza**: `URM_validation_train` per il training, `URM_validation_test` per il valutare (= ultimi 10% di `URM_train` per-utente).
- **Refit finale**: una volta scelti gli HP, viene fatto un re-fit su `URM_train_last_test = URM_train` (cioè il full train) e poi si valuta su `URM_test`.

> I best-HP risultanti per i tre dataset (DCCF) sono **hardcoded** in
> [`run_experiments_for_DCCF_original_baselines.py:84-100`](run_experiments_for_DCCF_original_baselines.py:84).
> La cifra che leggiamo nelle tabelle Shehzad si ottiene rieseguendo il fit con
> questi HP, **non** rifacendo la HP search.

---

## 8. Dipendenze e installazione

### 8.1 Dipendenze dichiarate

- `requirements_gpu.txt` (env principale, Python 3.8): torch 2.3.1+cu118, torch_geometric, torch-sparse/scatter/cluster, numpy 1.23.5, scipy 1.10.1, pandas 1.5.3, scikit-learn **0.24.2** (molto vecchio), scikit-optimize 0.8.1, pyyaml 5.4.1, tables 3.8.0, numba, ecc.
- `requirements_IntentAwareRS_DGCF.txt` (env separato, Python 3.6 + TF 1.14): per DGCF, oggi non installabile su macOS ARM64 senza acrobazie. Verrà rimosso in pulizia.

### 8.2 Cosa serve davvero alle baseline non-neurali

Solo numpy + scipy + pandas + scikit-learn + scikit-optimize (per il tuning), più alcuni utility (`prettytable`, `tqdm`, `pympler`, `tables`, ecc.). Vedi `repro_check_baseline.py` nella root, che dichiara l'env minimo via `.venv`.

### 8.3 Problemi di compatibilità incontrati

Sintetizzato (dettagli in `REPRODUCIBILITY_CHECK.md`):
- **NumPy 2.x** rimuove `np.in1d`, `np.float`, `np.int`, `np.bool`. Sul codepath
  non-neurale che usiamo:
  - `np.in1d` (`Evaluator.py:346`) → richiede numpy < 2. Soluzione: pinned a numpy 1.26.4.
  - `np.int` (`Compute_Similarity_Python.py:387`, `Compute_Similarity_Euclidean.py:192`) → richiede numpy < 1.24. Soluzione: **patch minima** (`np.int` → `int`), come raccomandato dalle release notes NumPy stesse; nessun cambio di logica.
- **Cython similarity**: il dispatcher `Compute_Similarity` ha un `try/except ImportError` per `from Recommenders.Similarity.Cython.Compute_Similarity_Cython import ...`. L'import path è quello vecchio (senza `topn_baselines_neurals.` prefix), quindi fallisce sempre e si torna automaticamente alla similarity Python. Per le nostre baseline è OK (un po' più lento ma ~1-2 min su Gowalla). Non patchiamo questa cosa: il codice di Shehzad gira così com'è.
- **scikit-learn 0.24.2** non è installabile su Python 3.11 ARM64. Soluzione: usiamo `scikit-learn ≥ 1.x` (i metodi `normalize` e simili usati dal codice non sono cambiati).

---

## 9. Aspetti rilevanti per il nostro modello FM situazionale (rimando a Fase 3)

- C'è già una cartella `Recommenders/FactorizationMachines/` (per `LightFMRecommender`,
  attualmente non importato in nessuno script attivo). Potrebbe essere riusata come
  punto di aggancio per integrare il nostro FM situazionale nel framework di
  Shehzad senza riscrivere la pipeline di evaluation.
- L'infrastruttura `BaseRecommender` richiede solo che il modello esponga
  `_compute_item_score(user_id_array, items_to_compute=None) → ndarray (n_users, n_items)`.
  Implementazioni semplici: il modello deve poter produrre uno score per ogni coppia
  (user, item). Per FM situazionale dovremo decidere come gestire il contesto
  durante l'evaluation (al momento del test non abbiamo un contesto naturale —
  da discutere).
- Il framework supporta `ICM_all` (Item Content Matrix) e `UCM_all` (User Content
  Matrix) come argomenti opzionali — `Recommenders/BaseCBFRecommender.py`. Sono
  attualmente passati come `None` per i dataset DCCF (`run_experiments_for_DCCF_original_baselines.py:45-46`),
  ma esiste l'infrastruttura per nutrire feature side-information se le costruiamo.
- Il `Data_manager/DatasetMapperManager` ha già il concetto di feature aggiuntive
  via `add_ICM(ICM_dataframe, "ICM_genres")` o `add_UCM(...)`. Riusabile per le
  feature contestuali se passiamo al Plan B/C (vedi `DATA_INVENTORY.md` §5).

---

## 10. Cheatsheet veloce (per chi entra nel repo)

```bash
# Setup env minimale (solo baseline non-neurali, no GPU)
python3 -m venv .venv
source .venv/bin/activate
pip install 'numpy<2' 'scipy<1.14' 'pandas<2.2' scikit-learn scikit-optimize tqdm dill prettytable pyyaml psutil pympler networkx tables numba

# Repro check rapida (solo baseline, no DCCF/BIGCF)
python repro_check_baseline.py --dataset gowalla --model all      # ~7 min su M2
python repro_check_baseline.py --dataset amazonBook --model all   # ~15 min
python repro_check_baseline.py --dataset tmall --model all        # ~15 min

# Risultati canonici di Shehzad (NON rieseguibili senza GPU per DCCF/BIGCF)
ls results/DCCF/      # ground-truth per il confronto
ls docs/tables_window/  # tabelle HTML — attenzione, contengono errori di doc

# Stato originale (immutato) e nostra branch pulita
git checkout original-shehzad-v1   # tag, sola lettura
git checkout clean-thesis-setup    # nostro lavoro
```
