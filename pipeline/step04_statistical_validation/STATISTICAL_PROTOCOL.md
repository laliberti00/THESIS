# STATISTICAL_PROTOCOL.md

Protocollo statistico per la Fase 2 della tesi. Documenta cosa fa
[`statistical_validation.py`](statistical_validation.py), perché lo fa così
e come si legge l'output.

Leggibile da chi non ha scritto il codice. Riferimenti incrociati a
[`SHEHZAD_PROTOCOL.md`](SHEHZAD_PROTOCOL.md) (cosa fa il protocollo di
Shehzad) e [`EVALUATOR_PATCH.md`](EVALUATOR_PATCH.md) (come abbiamo modificato
l'evaluator per rendere possibile questa analisi).

---

## 1. Sommario

| Cosa             | Scelta                                                                    |
| ---------------- | ------------------------------------------------------------------------- |
| Test puntuale    | Wilcoxon signed-rank, **paired**, `zero_method='pratt'`, two-sided        |
| IC sulla media   | **Percentile bootstrap** sulla media per-utente, `n_boot=10000`, α=0.05   |
| Modelli stocastici (DCCF, BIGCF, K=5 seed) | Wilcoxon per ciascun seed → 5 p-value → **Harmonic Mean P-value** (Wilson 2019, PNAS) come unico p-value combinato |
| Multiplicità     | **Holm step-down** applicato esclusivamente sui 4 confronti primari       |
| Soglia           | α = 0.05 familywise sul gruppo dei 4 primari                              |
| Metrica primaria | **Recall@20** (allineata a Shehzad: è la metrica che lui ottimizza in HP search) |
| Metrica secondaria | NDCG@20 — riportata sempre, non entra in Holm                           |
| Split            | Identico a Shehzad: `train.pkl` / `test.pkl` del DCCF split su Gowalla    |

---

## 2. Cosa è un "confronto primario"

In Fase 3 il modello sotto test sarà la nostra **Factorization Machine
situation-aware** (FM-situ). Lo confronteremo contro quattro avversari
in un'unica famiglia con controllo familywise α=0.05:

| # | Modello sotto test | Avversario          | Tipo del test                |
| - | ------------------ | ------------------- | ---------------------------- |
| 1 | FM-situ            | FM-flat (vanilla)   | paired Wilcoxon              |
| 2 | FM-situ            | best non-neural (la più forte di RP3β/ItemKNN/P3α/UserKNN/EASE^R) | paired Wilcoxon |
| 3 | FM-situ            | DCCF (5 seed)       | per-seed Wilcoxon → HMP      |
| 4 | FM-situ            | BIGCF (5 seed)      | per-seed Wilcoxon → HMP      |

I 4 p-value vengono corretti con la procedura di Holm. Un confronto
"rigetta H₀ a α=0.05 con familywise control" se il suo p-value
Holm-aggiustato è ≤ 0.05.

**In Fase 2 questi 4 confronti reali non sono ancora possibili** perché
(a) il modello situazionale non esiste, (b) DCCF/BIGCF non sono stati
addestrati (servono GPU). Lo script gira quindi su un **placeholder** che
mantiene la stessa struttura logica:

| # primario reale                                | Placeholder Phase 2                                    |
| ----------------------------------------------- | ------------------------------------------------------ |
| FM-situ vs FM-flat                              | RP3β vs ItemKNN                                        |
| FM-situ vs best non-neural                      | RP3β vs P3α                                            |
| FM-situ vs DCCF (5 seed)                        | RP3β vs UserKNN ×5 (Gaussian-noise stub σ=0.01)        |
| FM-situ vs BIGCF (5 seed)                       | RP3β vs TopPop ×5 (Gaussian-noise stub σ=0.005)        |

Lo scopo della Phase 2 è **esercitare la meccanica del test** (Wilcoxon
+ HMP + Holm + bootstrap CI), non produrre conclusioni scientifiche. Le
conclusioni arrivano in Phase 3 con il modello vero e i dati veri.

> **Nota sui Gaussian-noise stub.** Per i modelli stocastici (DCCF, BIGCF)
> in Phase 3 avremo 5 vettori per-utente reali, uno per seed. In Phase 2
> li simuliamo aggiungendo rumore i.i.d. gaussiano (zero-mean, σ piccola)
> al vettore per-utente di una baseline deterministica, poi clipping a
> [0,1]. Questo introduce un artefatto: gli zero (gli utenti senza hit nel
> top-K) diventano valori piccoli positivi, riducendo drasticamente la
> percentuale di tie. È esplicitato nel TSV descrittivo dalla colonna
> `mean_across_seeds=True`. Nel Phase 3 reale i tie torneranno.

---

## 3. Test puntuale: Wilcoxon signed-rank

### Perché Wilcoxon e non t-test

- **Non parametrico**: non assume distribuzione gaussiana delle metriche
  per-utente, e quelle non sono gaussiane (Recall e NDCG sono bound in
  [0,1], con accumulo di massa su 0). Il paired t-test, sotto
  asimmetria/zero-inflated distribution, dà p-value sottostimati e
  intervalli di confidenza poco calibrati.
- **Paired**: confrontiamo lo stesso utente sotto due trattamenti
  (modello A vs modello B). I dati sono per definizione apparati.
- **Two-sided**: non assumiamo a priori la direzione dell'effetto. È
  prassi standard nei paper SIGIR / RecSys per evitare bias di reporting.

### Contratto di pairing per `user_id` (regola di sicurezza)

> **Regola.** Prima di passare due vettori a `scipy.stats.wilcoxon`, lo
> script li allinea esplicitamente **per `user_id`**, non per posizione.
> Cioè per ogni indice `i` deve valere che `vec_A[i]` e `vec_B[i]` sono
> i valori della metrica per **lo stesso utente**.

Perché la regola è esplicita. Due `.npz` per-utente prodotti dallo
*stesso* `EvaluatorHoldout` sullo *stesso* `URM_test` condividono il
vettore `user_ids` nello stesso ordine — l'evaluator itera la lista
deterministica `self.users_to_evaluate`. Tutti i `.npz` della Phase 2
soddisfano questa condizione (verificato empiricamente sui 8 file
attualmente in `repro_check_results/per_user/`). In Phase 3, però, i
5 `.npz` di DCCF e i 5 di BIGCF possono venire da run su macchine /
processi diversi, e non è garantito che il filtro `min_ratings_per_user`
o eventuali utenti cold scartati siano identici fra seed. In quel caso
i due `user_ids` array sarebbero di lunghezze diverse o nella stessa
lunghezza ma in ordine diverso — passare i vettori così come sono al
test paired darebbe risultati silenziosamente sbagliati.

La funzione `align_two()` in [`statistical_validation.py`](statistical_validation.py)
implementa l'allineamento: fast-path quando `np.array_equal(a.user_ids,
b.user_ids)` (caso Phase 2), altrimenti riallinea entrambi i vettori
sull'intersezione dei due `user_ids` preservando l'ordine del pivot e
scartando gli utenti presenti in uno solo dei due risultati. Ogni
chiamata a `paired_wilcoxon` nello script passa per `align_two`. Anche
il ramo multi-seed (Gaussian-noise stub Phase 2 e i 5 seed reali in
Phase 3) opera sul vettore opponent **post-allineamento**, garantendo
che ogni Wilcoxon per-seed appai gli stessi utenti.

### `zero_method='pratt'` invece del default

Su Gowalla, circa il **54 % degli utenti ha Recall@20 = 0** (vedi
`EVALUATOR_PATCH.md` §5). Il default di `scipy.stats.wilcoxon` è
`zero_method='wilcox'`, che **scarta** le coppie con differenza zero
prima del ranking. Sotto la nostra densità di tie questo riduce
drasticamente la potenza statistica. Wilcoxon Pratt invece **mantiene le
coppie a differenza zero nel ranking** (assegna loro un rango medio) e
poi le esclude solo dalla statistica del test: è il modo moderno
raccomandato sotto tie diffusi (Pratt, *J. Amer. Stat. Assoc.*, 1959;
ripreso in Conover, *Practical Nonparametric Statistics*, 1999).

---

## 4. Multi-seed → Harmonic Mean P-value (Wilson 2019)

### Il problema

DCCF e BIGCF sono modelli stocastici (inizializzazione random + sampling
batch + dropout). Una singola run non è rappresentativa: serve raccoglere
risultati da K seed (qui K=5: 2022, 2023, 42, 0, 1). Per ciascun seed
otteniamo un vettore per-utente di Recall@20 → un Wilcoxon paired contro
il modello sotto test → un p-value `p_k`. Servono quindi 5 p-value
combinati in uno solo.

### Perché non la media per-utente sui seed

> **Decisione metodologica esplicita.** La media sui K seed del vettore
> per-utente — `r̄_u = mean_k r_u^{(k)}` — viene **scartata** come
> aggregazione del segnale stocastico.
>
> Maschera la varianza tra seed, gonfia artificialmente la significatività
> del test, e dà l'illusione di un effetto stabile dove invece c'è
> rumore: con il diminuire della varianza tra seed, il p-value tende a
> zero anche se ogni singola run è poco significativa. È una pratica
> bocciata in letteratura statistica (vedi anche Demšar 2006 JMLR per
> argomentazione equivalente sul confronto multi-classifier).

### Perché HMP e non Fisher

| Metodo                                       | Assume indipendenza? | Comportamento sotto dipendenza |
| -------------------------------------------- | -------------------- | ------------------------------ |
| **Fisher's combined probability** χ²=-2Σln pᵢ | Sì                  | Anticonservativo (rigetta troppo) |
| **Stouffer's Z**                              | Sì                  | Anticonservativo                |
| **Bonferroni (min-p × K)**                    | No (gestita assumendo worst case) | Molto conservativo  |
| **Harmonic Mean P-value (Wilson 2019)**       | No                   | **Calibrato sotto dipendenza arbitraria** |

I nostri K=5 test sono **non indipendenti**: stesso test set, stessi
utenti, stesso modello sotto test, stesso baseline. Le sorgenti di
dipendenza sono molteplici. Fisher e Stouffer non sono affidabili sotto
queste condizioni — il loro p-value combinato sarebbe ottimisticamente
piccolo. HMP, introdotto in [Wilson D.J. (2019), "The harmonic mean
p-value for combining dependent tests", PNAS 116:1195-1200], è
specificamente progettato per il nostro scenario.

### Implementazione

In [`statistical_validation.py:harmonic_mean_p`](statistical_validation.py):

```
HMP   = 1 / mean(1 / p_i)             # i = 1..L,  L = 5 seed
p_HMP ≈ HMP × e × ln(L)                # asintotico Wilson 2019, clip [0,1]
```

Il p-value combinato `p_HMP` è ciò che entra nel passo di Holm. La forma
asintotica è quella raccomandata da Wilson per L modesto (≤ ~20); è
deliberatamente conservativa.

---

## 5. Correzione di Holm sui 4 confronti primari

### Perché Holm e non Bonferroni

Bonferroni controlla il familywise error rate a costo di una grande
perdita di potenza (moltiplica ogni p-value per K). Holm (1979) ha la
**stessa garanzia di familywise control sotto qualunque dipendenza**, ma
è uniformly more powerful — non rigetta meno di Bonferroni e tipicamente
rigetta di più.

### Procedura

Sia `p_(1) ≤ p_(2) ≤ … ≤ p_(K)` la sequenza ordinata dei p-value della
famiglia (K = 4 nel nostro caso). Definiamo:

```
p_adj_(j) = max_{m ≤ j}  [ (K - m + 1) × p_(m) ]    ,  cap a 1
```

Rigettiamo H₀ del confronto j-esimo se `p_adj_(j) ≤ α = 0.05`.

### Perché applicarlo SOLO ai 4 confronti primari

La famiglia è definita **a priori**: i 4 confronti decisi prima di
guardare i dati. Non includiamo nel correctione, ad esempio, anche
confronti di sensitività (es. RP3β vs ItemKNN su NDCG@20, o su Recall@40),
perché farlo gonfia K artificialmente e riduce la potenza sui confronti
che ci interessano davvero. NDCG@20 e gli altri cutoff vengono comunque
riportati come descrittivi, ma fuori dalla famiglia Holm.

---

## 6. Bootstrap CI 95 % sulla media per-utente

### Percentile bootstrap

Per ogni modello, calcoliamo l'intervallo di confidenza al 95 % della
media per-utente di Recall@20 (e NDCG@20) **ricampionando con sostituzione
il vettore per-utente n_boot = 10000 volte** e prendendo il 2.5° e il
97.5° percentile della distribuzione delle medie ricampionate.

### Percentile vs BCa

Per n_user ≫ 1 (qui ~38k su Gowalla) e distribuzioni approssimativamente
simmetriche dopo aggregazione (la media è approssimativamente gaussiana
per CLT), percentile bootstrap è ben calibrato e ha il vantaggio di
essere immediato da spiegare. BCa (bias-corrected and accelerated) sarebbe
giustificato per distribuzioni fortemente asimmetriche o piccoli n, che
non è il nostro caso. Cfr. Efron & Tibshirani (1993), *An Introduction
to the Bootstrap*, §13.

---

## 7. Estensione rispetto al protocollo di Shehzad

Riassunto (vedi `SHEHZAD_PROTOCOL.md` per i dettagli):

| Aspetto                   | Shehzad (SIGIR 2025)        | Nostra Fase 2                                     |
| ------------------------- | --------------------------- | ------------------------------------------------- |
| Seed multipli su modelli stocastici | ❌ (1 seed singolo)  | ✅ (5 seed, HMP aggregato)                        |
| Test di significatività   | ❌ (medie senza test)      | ✅ Wilcoxon paired                                 |
| Multiplicità              | ❌                         | ✅ Holm su famiglia primaria                       |
| Intervalli di confidenza  | ❌                         | ✅ Bootstrap percentile 95 %                      |
| Per-user metrics          | ❌ (scartati)              | ✅ Esposti dall'evaluator patchato                 |

**Dichiarazione metodologica per la tesi**: la nostra valutazione
statistica **aggiunge rigore al protocollo di Shehzad et al. (2025)**.
La replica numerica delle baseline su Gowalla è stata fatta (vedi
[`REPRODUCIBILITY_CHECK.md`](REPRODUCIBILITY_CHECK.md)) e i numeri
coincidono entro 0.06 % su Recall@20. La procedura inferenziale (test,
IC, multiplicità) è ortogonale e va dichiarata in tesi come nostro
contributo metodologico, non come fedele riproduzione di Shehzad.

---

## 8. Come si legge l'output dello script

```bash
python statistical_validation.py \
    --dataset gowalla \
    --metric RECALL --cutoff 20 \
    --out-descr results_phase2/descriptive.tsv \
    --out-tests results_phase2/tests.tsv
```

### `descriptive.tsv` — una riga per modello

| Colonna             | Significato                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| `model`             | Etichetta del modello (es. "RP3β (pivot)")                                        |
| `n_users`           | Numero di utenti valutati (38 234 su Gowalla)                                     |
| `mean`              | Media del per-utente per la metrica e cutoff richiesti                            |
| `std`               | Deviazione standard campionaria (`ddof=1`)                                        |
| `ci95_lo`, `ci95_hi`| Estremi dell'IC bootstrap al 95 %                                                 |
| `pct_zero`          | Frazione di utenti con metrica = 0 (utile per leggere il peso dei tie nel Wilcoxon) |
| `mean_across_seeds` | `True` se il modello è uno stub multi-seed; `NaN` per le baseline deterministiche |

### `tests.tsv` — una riga per confronto primario

| Colonna              | Significato                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `comparison`         | Nome del confronto primario                                                              |
| `model_under_test`   | Pivot del confronto                                                                      |
| `opponent`           | Avversario                                                                               |
| `p_raw`              | Per i deterministici: p-value Wilcoxon two-sided. Per i multi-seed: p-value HMP combinato (= `e × ln(L) × HMP`) |
| `p_holm`             | Stesso p-value dopo Holm step-down sulla famiglia dei 4                                   |
| `reject_at_0.05`     | `True` se `p_holm ≤ 0.05`                                                                |
| `details_json`       | JSON con dettagli (per i multi-seed: i 5 p-value individuali e l'HMP grezzo)             |

### Esempio (placeholder su Gowalla, Recall@20)

Dalla run del 2026-05-24:

```
                                                 model  n_users     mean      std  ci95_lo  ci95_hi  pct_zero
                                          RP3β (pivot)    38234 0.245458 0.344263 0.242015 0.248866  0.539
                     ItemKNN (placeholder for FM-flat)    38234 0.235463 0.340246 0.232023 0.238815  0.556
                 P3α (placeholder for best non-neural)    38234 0.231430 0.337794 0.228035 0.234786  0.560
UserKNN ×5 (placeholder for DCCF, Gaussian-noise stub)    38234 0.217258 0.327129 0.214003 0.220514  0.018
TopPop ×5 (placeholder for BIGCF, Gaussian-noise stub)    38234 0.027201 0.119483 0.025997 0.028444  0.029
```

```
                           comparison         p_raw        p_holm  reject_at_0.05
      primary_1__vs_FM_flat_surrogate  4.34e-18      4.34e-18      True
primary_2__vs_best_baseline_surrogate  3.75e-42      7.51e-42      True
         primary_3__vs_DCCF_surrogate  2.92e-61      8.75e-61      True
        primary_4__vs_BIGCF_surrogate  4.37e-300     1.75e-299     True
```

Lettura: **tutti e 4 i confronti rigettano H₀** a α=0.05 dopo Holm.
Sensato: su Gowalla RP3β è uniformemente sopra le altre baseline; i
p-value piccolissimi riflettono il grande n (38k utenti) e una
differenza misurabile media-utente.

> **Attenzione**: questi sono valori *placeholder*, non risultati
> scientifici sulla tesi. In Phase 3 il pivot diventerà la nostra
> FM-situ e cambieranno tutti i numeri.

---

## 9. Roadmap di affinamento per la Fase 3

Quando arriveranno DCCF e BIGCF reali dalla macchina universitaria:

1. **5 seed di DCCF** (e di BIGCF). Per ciascuno, salvare il vettore
   per-utente di Recall@20 (e NDCG@20) come `.npz` nello stesso formato
   dei nostri placeholder. Convenzione naming proposta:
   `repro_check_results/per_user/gowalla_DCCF_seed{S}.npz`.
2. Sostituire nello script i blocchi "Gaussian-noise stub" con i 5
   `.npz` reali (parametro `opponent_npzs`).
3. Sostituire il "RP3β pivot" con il `.npz` della FM-situ una volta
   addestrata.
4. Mantenere identici tutti gli altri parametri statistici (test,
   correzione, IC, n_boot). Solo gli input cambiano.

La pipeline è già pronta per accettare un numero variabile di `.npz`
per opponent stocastico (vedi `Comparison.opponent_npzs`).

---

## 10. Riferimenti

- **Wilcoxon signed-rank con Pratt**: Pratt, J.W. (1959), "Remarks on
  Zeros and Ties in the Wilcoxon Signed Rank Procedures", *Journal of
  the American Statistical Association*, 54(287):655–667.
- **Harmonic Mean P-value**: Wilson, D.J. (2019), "The harmonic mean
  p-value for combining dependent tests", *PNAS*, 116(4):1195–1200.
  doi: 10.1073/pnas.1814092116
- **Holm step-down**: Holm, S. (1979), "A simple sequentially rejective
  multiple test procedure", *Scandinavian Journal of Statistics*,
  6(2):65–70.
- **Bootstrap percentile**: Efron, B. & Tibshirani, R. (1993), *An
  Introduction to the Bootstrap*, Chapman & Hall, ch. 13.
- **Critica della media-su-seed**: Demšar, J. (2006), "Statistical
  Comparisons of Classifiers over Multiple Data Sets", *Journal of
  Machine Learning Research*, 7:1–30 (argomentazione equivalente in
  contesto multi-dataset).
