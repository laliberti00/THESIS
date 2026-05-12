# PHASE1_SUMMARY.md

Memo finale di Fase 1 della tesi di dottorato in recommender systems.
Pensato per essere letto **fuori dal contesto del codice** — può circolare in
una chat di discussione senza che chi legge debba aprire il repo.

---

## TL;DR

- ✅ Repo originale di Shehzad capito a fondo, mappato e ridotto al perimetro
  di tesi (DCCF + BIGCF + baseline non-neurali tunate × Gowalla / AmazonBook / Tmall).
- ✅ Numeri di Shehzad **riprodotti su Gowalla** entro tolleranza ≤ 0.06 %
  per Recall@20 e ≤ 0.015 % per NDCG@20 (vedi [REPRODUCIBILITY_CHECK.md](REPRODUCIBILITY_CHECK.md)).
- ✅ Documentazione completa prodotta: `CODE_ANALYSIS.md`, `DATA_INVENTORY.md`,
  `REPRODUCIBILITY_CHECK.md`, `PHASE1_SUMMARY.md`, README aggiornato.
- ✅ Repo confezionato come progetto navigabile in VS Code, con env CPU
  funzionante e istruzioni separate per env GPU.
- 🟡 **Decisione bloccante per Fase 3**: i `.pkl` di Shehzad **non contengono
  alcun segnale contestuale**. Servono input da Luca per scegliere fra Plan A / B / C
  (vedi §3 di questo documento).

---

## 1. Cosa è stato fatto

1. **Inizializzazione git** del repo originale, tag `original-shehzad-v1` sullo
   stato pristino, branch di lavoro `clean-thesis-setup`. La cartella originale
   (`~/Downloads/IntentAwareRS_original/`) resta come back-up di sicurezza.
2. **Ispezione empirica** di tutti i `.pkl` di Shehzad via `inspect_pkls.py`:
   formato, contenuti, statistiche per ciascun dataset. Tutto in
   [DATA_INVENTORY.md](DATA_INVENTORY.md).
3. **Lettura sistematica** del codice del framework `topn_baselines_neurals/`:
   loader dataset, splitting train/val/test, evaluator, metriche, HP search,
   modelli baseline e deep. Tutto in [CODE_ANALYSIS.md](CODE_ANALYSIS.md).
4. **Setup env CPU minimale** (venv Python 3.11, numpy<2, scipy, pandas,
   sklearn, scikit-optimize, ecc.) bypassando la pesante dipendenza torch+CUDA
   che non serve per le baseline.
5. **Patch minimali di compatibilità** (`np.int → int` in due file Cython-less
   della similarity). Stesso esatto fix che le release notes NumPy raccomandano,
   nessun cambio di logica.
6. **Reproducibility check su Gowalla** per le 4 baseline tunate (RP3β, ItemKNN,
   UserKNN, P3α) più i due baseline non-tunati (TopPop, Random). 4 baseline su
   4 passano la soglia di 1 %, 2 sono *bit-identiche*. Random è noise, non
   significativo.
7. **Confezionamento di una repo pulita standalone** in `~/Downloads/IntentAwareRS_thesis/`
   con:
   - solo i 3 dataset di tesi (Gowalla, AmazonBook, Tmall),
   - solo i modelli di tesi (DCCF, BIGCF, 5 baseline non-neurali),
   - script CLI: `repro_check_baseline.py` (CPU), `run_experiments_for_DCCF_original_baselines.py`,
     `run_experiments_for_BIGCF_original.py`, `run_hyperparameter_search_baseline_DCCF_for_datasets.py`,
   - `requirements-cpu.txt` + `requirements-gpu.txt` separati,
   - `setup_env.sh` one-shot bootstrap,
   - README operativo + 4 markdown di documentazione.

## 2. Cose che ho trovato di rilevante (sorprese)

### 2.1 Sorpresa critica: nessun contesto nei dati di Shehzad

I `.pkl` sono `scipy.sparse.coo_matrix` con `data` uniformemente = 1.0. Solo
`(user_index, item_index)`. Niente timestamp, niente location (Gowalla),
niente categoria/brand/action_type (Tmall e AmazonBook nelle versioni raw
hanno queste feature), niente sessione, niente demografia, niente mapping
verso gli ID originali del raw dataset.

Conseguenza: **non è possibile costruire un FM situation-aware lavorando
direttamente sugli split di Shehzad**. La sola scelta che si avrebbe dentro
gli split sarebbe modellare `(user, item)` come un classico FM degenerato.

Questa è la decisione bloccante per la Fase 3, vedi §3.

### 2.2 Discrepanze fra documentazione HTML e codice / dati

Importante per la sezione *Methodology* del paper di tesi: **se Shehzad ha
errori interni di documentazione, dobbiamo basarci sui file `results/DCCF/*.txt`
come ground truth, non sulle tabelle HTML**. Esempi concreti:

1. **`docs/tables_window/tables_window_DCCF.html` — sezione Gowalla**:
   - Dice "Items = 41 390", ma le matrici sparse hanno shape `(50821, 57440)`.
     Il valore 41 390 è quello di Tmall, copia-incollato per errore.
   - Dice "Interactions per user min = 5", ma il `train.pkl` di Gowalla ha
     utenti con 1 sola interazione di training. Forse un filtro 5-core era
     dichiarato in preprocessing ma non rispettato bit-perfect dal `.pkl`
     finale.
   - La tabella *Optimal hyperparameter values* per Gowalla mostra
     `RP3β: topK=496, α=0.4447..., β=0.5968..., normalize=True`. Ma il
     **codice** `run_experiments_for_DCCF_original_baselines.py:87-88`
     hardcoda per Gowalla `RP3β: topK=777, α=0.5664, β=0.001085, normalize=True`.
     I valori HTML sono quelli di AmazonBook copia-incollati. Il
     codice è quello che ha generato la R@20=0.245 riportata, quindi è autoritativo.

2. **`AmazonBook/valid.pkl` esiste ma non viene mai letto**. Il loader
   `Gowalla_AmazonBook_Tmall_DCCF.py:35-37` apre solo `train.pkl` e `test.pkl`.
   La validazione è ricostruita sinteticamente con un sub-split per utente del
   `URM_train` (default `validation_portion=0.1`), uguale per tutti e tre i
   dataset. Se vogliamo restare numericamente equivalenti, **non dobbiamo
   usare il `valid.pkl`** di AmazonBook.

3. **ItemKNN/UserKNN — feature_weighting='TF-IDF'**. Il codice di evaluation
   passa solo `topK` e `similarity`; tutti gli altri parametri (`shrink=1000`,
   `normalize=True`, `feature_weighting='TF-IDF'`) vengono dai default di
   `ItemKNNCFRecommender.fit()`. Il `feature_weighting='TF-IDF'` non è
   menzionato in nessun docs HTML ma viene effettivamente applicato. È
   un'informazione metodologicamente non banale da mettere nel paper.

4. **Validation split non è temporale**. Nel sub-split per validazione, l'ordine
   in `URM.indices` dopo CSR è **item_id crescente**, non chronological. Quindi
   il "10 % finale" mandato in validazione è l'ultimo 10 % per indice item, non
   per tempo. Vedi [CODE_ANALYSIS.md §5.3](CODE_ANALYSIS.md).

### 2.3 Vincoli computazionali emersi

- **EASE^R non gira su 16 GB di RAM**: richiede una Gram matrix densa
  `n_items × n_items` = 26 GB per Gowalla, 14 GB per Tmall, 48 GB per AmazonBook.
  Va eseguito sulla macchina universitaria.
- **DCCF/BIGCF richiedono CUDA**. Non eseguibili sul laptop, da fare in università.
- **Baseline non-neurali** sono comodissime su CPU: ~ 1-2 min per modello su
  Gowalla, ~ 10-15 min su AmazonBook/Tmall. Fattibile sul laptop in mezza
  giornata.

### 2.4 Note di compatibilità

- Il framework usa API NumPy pre-2.0 (`np.in1d`, `np.float`, `np.int`, `np.bool`).
  Su NumPy 1.26 solo `np.int` rompeva nel code path che ci interessa (KNN);
  patchato con `np.int → int`, equivalente semanticamente per definizione di
  NumPy stesso.
- `scikit-learn 0.24.2` pinnata da Shehzad non build su Python 3.11. Usiamo
  sklearn 1.x: il codice usa solo `sklearn.preprocessing.normalize`, stabile
  fra le due versioni.
- Cython similarity: l'import path nel codice di Shehzad
  (`from Recommenders.Similarity.Cython...`) è quello vecchio senza il prefix
  `topn_baselines_neurals.`, quindi `ImportError` silenzioso e fallback alla
  Python implementation. Funziona, è 2-3× più lento, irrilevante per Phase 1.

## 3. Decisioni che ho dovuto prendere autonomamente e perché

1. **Rimozione totale di DGCF, KGIN, IDS4NR e dei loro dataset**. Esplicitamente
   confermato da Luca dopo Checkpoint 1.
2. **EASE^R skippato in Phase 1**. RAM insufficiente sul laptop; non c'erano
   alternative se non ridimensionare la matrice. Rinviato alla macchina GPU.
3. **AmazonBook e Tmall non eseguiti in Phase 1**. Decisione concordata con
   Luca durante l'esecuzione: la reproducibility check su Gowalla è già
   sufficiente come segnale per la Phase 1, AmazonBook e Tmall si lanceranno
   contestualmente alla disponibilità GPU per DCCF/BIGCF.
4. **Patch `np.int → int`**. Decisione presa senza checkpoint: NumPy stesso
   raccomanda esattamente questo come substitution semanticamente neutra. Non
   è un cambio di logica del modello, è una correzione di alias rimossi.
5. **`LightFM` (`Recommenders/FactorizationMachines/`) trattenuto**, anche se
   non usato dagli script di Shehzad. È un FM e potrebbe servirci come baseline
   in Phase 3. Costa zero tenerlo, ridurlo per allineare scope è prematuro.
6. **Documentazione dei docs HTML come *non* fonte autoritativa**. Visto le
   discrepanze trovate (§2.2), ho deciso che in tesi useremo `results/DCCF/*.txt`
   come riferimento numerico, non le tabelle HTML.

## 4. Implications: Plan A / B / C per il modello FM situazionale

Tre opzioni che bisogna scegliere prima di iniziare la Phase 3.

### Plan A — Stiamo sugli split di Shehzad, modello FM senza contesto

Si lavora direttamente sui `.pkl` esistenti. Il "modello situation-aware FM"
si riduce a un FM su `(user, item)`, equivalente in pratica a una factorization
matriciale standard. Il vantaggio è la confrontabilità numerica con i risultati
di Shehzad e di tutta la letteratura DCCF/BIGCF. Lo svantaggio è che **si perde
l'intero argomento di tesi**: senza contesto non c'è "situation-aware". Plan A
ha senso solo come early-stage sanity check (verificare che il nostro FM funzioni
in linea con i baseline) o come ablation baseline nel paper finale.

### Plan B — Si ricostruisce il dataset dai raw originali

Si scaricano i raw dataset (SNAP Gowalla, Amazon Reviews 2018, IJCAI-15 Tmall),
si applica un preprocessing che riproduca **bit-by-bit** quello di Shehzad
(stessi filtri 5/10/20-core, stessa selezione utenti/item) e si ottengono split
arricchiti con timestamp / location / categoria / action_type. Lo script di
preprocessing di Shehzad **non è nel repo**, quindi va ricostruito leggendo i
paper originali di DCCF/BIGCF (che probabilmente specificano i filtri). C'è un
rischio non trascurabile di non-riproducibilità bit-perfect: anche differenze
minori nello user/item universe rendono i numeri non più direttamente
confrontabili con le sue tabelle. Plan B è la versione "ambiziosa": dà al
modello le feature contestuali ma rischia di rompere la confrontabilità.

### Plan C — Plan B + ri-tuning onesto dei baseline

Si fa Plan B (ricostruzione raw + feature contestuali) e in più si **ri-tunano
tutte le baseline non-neurali** sul nostro split, con la stessa Bayesian search
di Shehzad. Nel paper di tesi si riportano i nuovi numeri baseline come fonte
primaria, e si cita Shehzad come riferimento storico. Costo computazionale
maggiore (la HP search delle baseline è 100 trials × 5 modelli × 3 dataset =
qualche giorno macchina) ma metodologicamente è la scelta più pulita: tutti
i numeri sullo stesso split, tutte le baseline tunate in modo identico al nostro
modello. Plan C è quello che farei in un mondo ideale; la domanda è se il
costo + il delay di scheduling si giustifica per la tesi.

> **Da decidere prima di Phase 3.** La mia proposta tentativa è Plan C, ma è
> una decisione che richiede valutare il budget compute della macchina
> universitaria e la timeline di tesi. Posso aiutare a stimare più precisamente
> il costo se serve.

## 5. Punti da discutere prima di passare alla Fase 2

In ordine di priorità.

1. **Sciogliere Plan A/B/C.** Senza questa scelta non si fa Fase 3, ma soprattutto
   non si sa **se** la Fase 2 (protocollo statistico) deve poi essere ri-applicata
   a un nuovo split (Plan B/C) — oppure no, se restiamo su Plan A.
2. **Quale macchina GPU userai in università, con quanta RAM?** Determina la
   fattibilità di EASE^R su AmazonBook (48 GB Gram), e l'effort necessario
   per multiple seed di DCCF/BIGCF in Phase 2.
3. **Vogliamo aggiungere modelli intent-aware oltre a DCCF/BIGCF?** Es. DGCF
   stesso (anche se richiede env Python 3.6) o altri SOTA recenti.
   Probabilmente no — manteniamo lo scope.
4. **Quali metriche reporteremo in tesi?** Shehzad reporta R@20, R@40, NDCG@20,
   NDCG@40. È sufficiente o aggiungiamo anche metriche di diversità / fairness
   (l'evaluator le supporta ma sono commentate)?
5. **Per la Phase 2 (protocollo statistico):** quanti seed per DCCF/BIGCF
   (5? 10?), quale test statistico (paired t-test? Wilcoxon? Quade?), su quali
   coppie di modelli (FM-vs-baseline-migliore, FM-vs-DCCF, FM-vs-BIGCF, tutti)?
6. **Per quando hai la macchina GPU**, vuoi che produca uno script
   `run_phase2.sh` che lancia tutto in batch (5 seed × 2 deep model × 3 dataset,
   più baseline una volta sola) e raccoglie i numeri in una tabella unica?

## 6. Stato dei deliverable richiesti

| Deliverable                        | Stato      | File                                   |
| ---------------------------------- | ---------- | -------------------------------------- |
| 1. CODE_ANALYSIS.md                | ✅ Done    | [CODE_ANALYSIS.md](CODE_ANALYSIS.md)             |
| 2. REPRODUCIBILITY_CHECK.md        | ✅ Done    | [REPRODUCIBILITY_CHECK.md](REPRODUCIBILITY_CHECK.md) |
| 3. Branch pulito                   | ✅ Done    | nuova repo `IntentAwareRS_thesis/`     |
| 4. DATA_INVENTORY.md               | ✅ Done    | [DATA_INVENTORY.md](DATA_INVENTORY.md)             |
| 5. PHASE1_SUMMARY.md               | ✅ Done    | questo file                             |
| 6. README aggiornato               | ✅ Done    | [README.md](README.md)                   |
| 7. Codice eseguibile post-pulizia  | ✅ Done    | `repro_check_baseline.py` testato       |
