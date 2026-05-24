# EVALUATOR_PATCH.md

Patch additiva applicata a `topn_baselines_neurals/Evaluation/Evaluator.py`
per esporre, su richiesta, le metriche per-utente necessarie ai paired
statistical test della Fase 2.

> **Riassunto in una riga.** Aggiunto un flag opzionale `save_per_user=False`
> al costruttore di `Evaluator` / `EvaluatorHoldout`; quando vale `True` vengono
> salvate cinque metriche per ciascun utente (`PRECISION, RECALL, NDCG, MAP,
> MRR`) a tutti i cutoff. Quando vale `False` (default) **il comportamento è
> bit-identico a quello upstream di Shehzad**, verificato da un test
> automatico in `tests/test_evaluator_patch_equivalence.py`.

---

## 1. Perché la patch

Come documentato in [`SHEHZAD_PROTOCOL.md`](SHEHZAD_PROTOCOL.md) §Q4,
l'`EvaluatorHoldout` originale **non salva le metriche per utente**: vengono
accumulate via `+=` su uno scalare (PRECISION, RECALL, NDCG) o tramite
oggetti accumulator con solo `cumulative_*` e `n_users` (MAP, MRR), e i
valori individuali sono scartati. Per i paired test (Wilcoxon signed-rank
sui vettori di Recall per-utente di due modelli) servono i vettori
individuali, quindi senza una modifica al framework i test sono inapplicabili.

Vincolo: gli aggregati esistenti devono restare **numericamente identici**
prima e dopo la patch, perché tutti i numeri pubblicati di Shehzad in
`results/DCCF/*.txt` e i confronti che noi stessi abbiamo fatto in
[`REPRODUCIBILITY_CHECK.md`](REPRODUCIBILITY_CHECK.md) dipendono da quella
logica esatta.

---

## 2. Cosa è cambiato (cosa NON è cambiato)

Tutte le modifiche sono in un singolo file:
[`topn_baselines_neurals/Evaluation/Evaluator.py`](topn_baselines_neurals/Evaluation/Evaluator.py),
marcate da commenti `# === THESIS PATCH (Phase 2) ===`.

**Aggiunto** (additivo):
- Import di `average_precision` e `rr` da `metrics` (servono per calcolare il
  valore per-utente di MAP e MRR coerentemente con quanto fa l'evaluator
  esistente — sono esattamente le stesse funzioni che `MAP._Metrics_Object` e
  `MRR._Metrics_Object` chiamano internamente).
- Attributo di classe `PER_USER_METRICS = ("PRECISION", "RECALL", "NDCG", "MAP", "MRR")`.
- Parametro `save_per_user=False` nei costruttori di `Evaluator` e
  `EvaluatorHoldout`. Default `False` → identico comportamento upstream.
- In `evaluateRecommender`: inizializzazione *lazy* dei contenitori per-utente
  (Python list per append O(1), convertite in `np.ndarray` alla fine).
- In `_compute_metrics_on_recommendation_list`: l'append per-utente avviene
  **leggendo i return value scalari delle funzioni già chiamate dalla logica
  aggregata** (`precision()`, `recall()`, `ndcg()`), in modo che non si
  rifaccia mai il calcolo due volte e non ci sia drift numerico.
  Per MAP/MRR — i cui valori per-utente non sono esposti dagli accumulator
  esistenti — si chiama una volta in più `average_precision(is_relevant_cutoff)`
  e `rr(is_relevant_cutoff)`: sono pure functions, deterministiche, non hanno
  side effect e producono lo stesso valore che gli accumulator usano
  internamente.

**Non cambiato**:
- Logica di accumulo aggregato (`+=` per PRECISION/RECALL/NDCG,
  `.add_recommendations()` per MAP/MRR/NOVELTY/COVERAGE_ITEM).
- Divisione per `_n_users_evaluated` alla fine (riga ~282 del file).
- Costruzione del `pd.DataFrame` `results_df` e della stringa `results_run_string`.
- Firma e tipo di ritorno di `evaluateRecommender` (`results_df, results_run_string`).
- Comportamento con `save_per_user=False` (è il default → garantito invariante).

---

## 3. Come consumare i per-utente

```python
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout

evaluator = EvaluatorHoldout(
    URM_test,
    cutoff_list=[1, 5, 10, 20, 40, 50, 100],
    exclude_seen=True,
    save_per_user=True,             # <-- ON
)
results_df, results_str = evaluator.evaluateRecommender(recommender)

# Aggregato (uguale a prima della patch):
print(results_df.loc[20, "RECALL"])  # e.g. 0.245458

# Per-utente (nuovo):
recall_20 = evaluator.per_user_metrics[20]["RECALL"]   # np.ndarray (n_eval,)
ndcg_20   = evaluator.per_user_metrics[20]["NDCG"]
user_ids  = evaluator.per_user_user_ids                # np.ndarray (n_eval,)

# Invariante garantito:
assert abs(recall_20.mean() - results_df.loc[20, "RECALL"]) < 1e-12
```

### Metriche esposte per-utente

`PRECISION`, `RECALL`, `NDCG`, `MAP`, `MRR`.

Non sono esposte per-utente:
- `F1` — è una funzione *derivata* di `PRECISION` e `RECALL`; recalcolabile
  on-the-fly come `2·p·r/(p+r)` da un vettore di per-utente.
- `NOVELTY`, `COVERAGE_ITEM` — sono globalmente aggregati (`NOVELTY` dipende
  dalle popolarità del train set e somma sulle liste raccomandate;
  `COVERAGE_ITEM` è un set di item raccomandati su tutta la base utenti).
  Per definizione non hanno un valore per-utente unico.

---

## 4. Test di equivalenza

File: [`tests/test_evaluator_patch_equivalence.py`](tests/test_evaluator_patch_equivalence.py).
Verifica tre invarianti su Gowalla con RP3β e gli HP best di Shehzad
(`topK=777, alpha=0.5664, beta=0.001085, normalize=True`):

| Invariante                                              | Tolleranza | Risultato                                  |
| ------------------------------------------------------- | ---------- | ------------------------------------------ |
| **I1** — aggregate con `save_per_user=False` == aggregate con `save_per_user=True` | `1e-12` | ✅ tutti i 56 (cutoff × metrica) pair bit-identici |
| **I2** — aggregate con `save_per_user=True` ≈ Shehzad reference (`results/DCCF/gowalla_RP3betaRecommender.txt`) | `1e-3` (10× più stretto dell'1 % concordato) | ✅ max relative deviation = `6.4e-4` |
| **I3** — `mean(per_user_metrics[cutoff][metric]) == aggregate(cutoff, metric)` | `1e-12` | ✅ tutti i 35 (cutoff × metrica) pair |

**Esecuzione** dalla root del repo:

```bash
source .venv/bin/activate
python -m tests.test_evaluator_patch_equivalence
```

Costa ~1 min su MacBook Air M2: load Gowalla 9 s + fit RP3β 33 s + 2 × eval
26 s.

**Overhead della patch** (cost ratio `save_per_user=True / False`): 27.3 / 25.4 =
**+7.5 %** wall-clock. Praticamente irrilevante per il nostro use case.

---

## 5. Statistiche utili per il design del test statistico

Misurate sulla run di RP3β / Gowalla / cutoff = 20:

| Metric    | n_users | mean    | std     | min | max | % a zero |
| --------- | ------- | ------- | ------- | --- | --- | -------- |
| RECALL@20 | 38234   | 0.2455  | 0.3443  | 0   | 1   | **53.9 %** |
| NDCG@20   | 38234   | 0.1486  | 0.2205  | 0   | 1   | **53.9 %** |

Più della metà degli utenti ha Recall@20 = 0 (cioè zero hit nei top-20). È un
tasso di tie altissimo per i paired test: nello script di validazione
statistica useremo `scipy.stats.wilcoxon(..., zero_method='pratt')` invece
del default `'wilcox'` (che scarta i tie e riduce la potenza statistica).
Dettaglio documentato in [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md).

---

## 6. Compatibilità verso il futuro

- La patch è **additiva** rispetto all'upstream `RecSys2019_DeepLearning_Evaluation`.
  Una eventuale futura merge da upstream Shehzad richiederà di reapplicare
  questo singolo file; tutti i blocchi sono delimitati da
  `# === THESIS PATCH (Phase 2) ===`/`# ==…` per identificarli.
- I `.txt` di output del run dell'eval finale (`results/DCCF/*.txt`) **non
  cambiano formato**, perché la stringa `results_run_string` e il DataFrame
  `results_df` sono prodotti dalla stessa logica di prima.
- Il flag `save_per_user` è opt-in, quindi tutti gli script Shehzad esistenti
  (`run_experiments_for_DCCF_original_baselines.py`, `run_experiments_for_BIGCF_original.py`,
  `run_hyperparameter_search_baseline_DCCF_for_datasets.py`) continuano a
  funzionare invariati.
