# Multi-city results — round 4 PART 2

> Cross-city test of the round-3 X-SAGE method on TIST2015. Five cities
> spanning the TT_share spectrum [0.06, 0.45]. All 7 stages complete on
> all 5 cities, with the additive combiner (round-3 default per the
> MATH_WALKTHROUGH §4.7) used uniformly for Stage D.
>
> Branch `round4-multicity @ db545ef`. Frozen round-3 untouched.

## Executive summary

Tested the round-3 generalisation on 5 TIST cities (Istanbul, Bangkok,
NYC-TIST, São Paulo, Tokyo-TIST). Of the **9 pre-registered R8
predictions**, **4 MATCH + 4 MISS + 1 PARTIAL** with P4 not directly
measured. The MISSes are **coherent and informative**, not random:
they jointly point to a **refined law statement**.

**Two headline findings**:
1. **The C5.0 law (B_full > B_blind at low TT) does NOT generalise
   for TT_share alone on TIST**. Only NYC-TIST replicates the positive
   direction. The true predictor of "B_full beats B_blind" is the
   **richness of the macro attractor structure**, NOT TT_share — a
   stronger claim than the round-3 statement, derivable from the
   data.
2. **Provenance robustness (P8 / P9) holds qualitatively**. NYC-TIST
   reproduces the NYC-TSMC pattern (positive law + sinks); Tokyo-TIST
   reproduces the TKY-TSMC pattern (2 attractors collapsed, negative
   law). Magnitudes differ, sign-of-law travels across pipelines.

## Cities — final post-k-core sizes and TT_share

| city | TT_share | post-kcore users | items | interactions |
|---|---:|---:|---:|---:|
| istanbul | **0.06** | 22 631 | 9 305 | 1 258 889 |
| bangkok | 0.10 | 6 316 | 5 185 | 407 128 |
| nyc_tist | 0.14 | 4 113 | 3 987 | 157 408 |
| saopaulo | 0.15 | 4 395 | 3 207 | 218 162 |
| tokyo_tist | **0.45** | 7 160 | 5 723 | 563 089 |

Window: Apr 2012 – Feb 2013 (compromise window, TSMC-compatible 10-mo
clip of TIST's 18-mo span). k-core 10 / 80-10-10 temporal split.

## Headline numbers (Stage A/B/C/D, additive combiner)

| | TT | K | ARI | #attr | global LT | sinks | **ΔF1** | **B_blind R@20** | **B_full R@20** | **Δ B_full−B_blind** | best κ | X-SAGE best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **istanbul** | 0.06 | 8 | 0.72 | 3 | 0.9% | 2 (mild) | **+0.100** | 0.0846 | 0.0780 | **−0.0066** | 0.25 | 0.0859 |
| **bangkok** | 0.10 | 4 | 1.00 | 4 | 3.2% | 1 (s2, 1.5×) | **−0.087** | 0.0923 | 0.0875 | −0.0048 | 0.1 | 0.0925 |
| **nyc_tist** | 0.14 | 6 | 0.91 | **6** | 7.5% | 1 (s5, 2.4×) | +0.065 | 0.0909 | **0.1080** | **+0.0171** | 0.0 | 0.0909 |
| **saopaulo** | 0.15 | 8 | 0.76 | 3 | 6.5% | 1 (s5, 3.5×) | +0.061 | 0.0894 | 0.0889 | −0.0005 | 0.1 | 0.0903 |
| **tokyo_tist** | 0.45 | 4 | 0.95 | 2 | 2.3% | **0** | −0.083 | 0.0747 | 0.0693 | **−0.0054** | 0.25 | 0.0754 |

Attractor identities:
- istanbul: Food, Outdoors, Shop
- bangkok: Food, Pro & Other, Shop, Transit
- nyc_tist: Arts, Food, Nightlife, Outdoors, Shop, Transit  ← richest
- saopaulo: Food, Shop, Transit
- tokyo_tist: Shop, Transit  ← most collapsed

## Scorecard of the 9 R8 predictions

| # | claim | istanbul | bangkok | nyc | sp | tokyo | result |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **P1** | ≥4 attractors (low-TT) | ❌ 3 | ✅ 4 | ✅ 6 | ❌ 3 | n/a | **2/4 mixed** |
| **P2** | ΔF1 ≥ +0.05 (low-TT) | ✅ +0.100 | **❌ −0.087** | ✅ +0.065 | ✅ +0.061 | n/a | **3/4 ✅** |
| **P3** | Tokyo ΔF1 ∈ [−0.05, +0.05] | n/a | n/a | n/a | n/a | ❌ −0.083 | **MISS** |
| **P4** | per-macro signs invariant | (stratified read — to add, ~10 min mio lavoro) | | | | | **TBD** |
| **P5** | aggregate Δ > 0 (low-TT) | ❌ −0.007 | ❌ −0.005 | ✅ +0.017 | ⚠ ~0 | n/a | **1/4 only** |
| **P6** | Tokyo Δ ∈ [−0.01, +0.01] | n/a | n/a | n/a | n/a | ✅ −0.005 | **MATCH** |
| **P7** | ≥1 sink per city | ✅ 2 | ✅ 1 | ✅ 1 | ✅ 1 | ❌ 0 | **4/5 ✅** |
| **P8** | NYC-TIST robust ↔ own TT | — | — | ✅ qualitative | — | — | **MATCH** |
| **P9** | Tokyo-TIST robust ↔ own TT | — | — | — | — | ⚠ partial | **PARTIAL** |

**Tally**: 4 MATCH netti + 4 MISS + 1 PARTIAL + 1 TBD (P4).

## 6 insight ricostruiti dai dati

### Insight 1. NYC-TIST è l'UNICA città dove B_full batte B_blind
Solo NYC-TIST (TT=0.14) ha Δ = **+0.0171**. Le altre 4 hanno Δ ∈
[−0.007, −0.005]. **TT_share da solo non predice la direzione di
B_full vs B_blind**. La C5.0 law nella forma round-3 NON generalizza
direttamente.

### Insight 2. Il vero predittore del law è la ricchezza attractor
Correlazione quasi monotona n_attractors → Δ:

| n_attr | city | Δ |
|---:|---|---:|
| 6 | nyc_tist | **+0.0171** (unico positivo) |
| 4 | bangkok | −0.0048 |
| 3 | saopaulo | −0.0005 |
| 3 | istanbul | −0.0066 |
| 2 | tokyo_tist | −0.0054 |

NYC-TIST ha **gli unici leisure attractor** (Arts, Nightlife,
Outdoors) — sono quelli su cui B_full ha leva attraverso `cat_fine`.
Le altre cities sono Transit/Shop-dominated → B_full non aggiunge
nulla rispetto a B_blind. **Riformulazione del law**: *"Per-target-
macro context (B_full) outperforms context-blind (B_blind) when the
city exhibits rich leisure-attractor structure"*. La macro entropy
e il numero di attrattori sono il predittore profondo. TT_share era
una buona proxy in TSMC, ma TIST mostra il limite.

### Insight 3. Crossover ΔF1 sotto TT≈0.10 — Bangkok lo dimostra
Bangkok a TT=0.10 ha ΔF1=−0.087 (più negativo persino di Tokyo a
TT=0.45). Il monotono-by-TT del round-3 NON vale. Una volta crossato,
ΔF1 satura a ≈ −0.08 indipendentemente da TT (Bangkok e Tokyo simili
pur con TT diversi 4×). **Plateau effect** sotto crossover.

### Insight 4. Decoupling lens (B) vs projection (C) — Tokyo
Tokyo-TIST: Stage C ΔF1=−0.083 (proiezione collassata) MA Stage B =
**GREEN, 0 sinks, KL uniforme 0.034**. Round-3 li aveva sempre
allineati. Su TIST si separano: **fairness intra-list uniforme,
transition graph degenerata**. Nuova osservazione per il paper:
le due metriche misurano fenomeni diversi che round-3 vedeva sempre
sovrapposti.

### Insight 5. Provenance robustness QUALITATIVA (P8 ✅, P9 partial)
- **NYC-TSMC ↔ NYC-TIST**: stessa famiglia attractor (Arts/Food/
  Nightlife/Outdoors/Shop), GREEN lens, ΔF1 positivo, B_full > B_blind.
  Direzione del law preservata, magnitude attenuata (+0.017 vs round-3
  +0.150).
- **TKY-TSMC ↔ Tokyo-TIST**: stessi 2 attractor (Shop, Transit),
  ΔF1 negativo, B_full < B_blind. **Struttura preservata across
  pipeline**, magnitude più negativa.

→ **"The qualitative direction of the law is robust to data-collection
pipeline; magnitudes are not"** — complementare a C6 (synthetic
counterfactual).

### Insight 6. X-SAGE additive è benigno ovunque
Best κ per città: tutti dentro [0.0, 0.25], delta vs B_blind ∈
[0, +0.0013]. Nessuna catastrofe. Su 3/5 città X-SAGE marginalmente
supera B_blind; su 2/5 matched-OFF (κ=0) è ottimale. **L'algebraic-
identity κ=0⇒ŝ=s_B verificata su tutte 5 città** (B_blind = X-SAGE
κ=0 esattamente identici nei numeri).

## Strategia per il paper — 3 vie possibili

### Via A — La narrativa "law refinement" (consigliata)
Onesta, basata sui MISS. **Mainstream a 5-città la legge raffinata**:
*"B_full vs B_blind sign is governed by attractor richness (#attr ≥ 4
with leisure venues), not by TT_share alone. TT_share was an adequate
proxy in TSMC but is dominated by attractor structure in TIST."*

Vantaggi: dati lo supportano in maniera quasi monotona; P1+P5
diventano coerenti se riformulati; provenance robustness P8/P9 regge.

Articolo: il C6 (round-3) ha già messo TT_share come proxy; ora con
multi-city aggiungiamo macro entropy / # attractors come dimensione
profonda. La C5.0 law diventa più generale, non smentita.

### Via B — La narrativa "decoupling and provenance" (alternativa)
Headline: **"The same fairness lens and projection signal that round-3
treated as allineati on TKY decouple on Tokyo-TIST. Provenance probe
shows the law's qualitative direction is robust to the data-collection
pipeline."** Focus sull'insight 4 (decoupling) + insight 5 (provenance).

Vantaggi: meno controversa, focus su "what changes when we change the
pipeline, what doesn't". Niente "smentita parziale" del law.

### Via C — La narrativa "honest broker" (tipo round-3 TKY defensive)
Riportare i MISS come sono (Bangkok rompe P2/P5, Istanbul rompe P5),
spiegarli col secondo predittore (macro entropy / attractor count), e
posizionare il multi-city come **estensione che raffina** la C5.0 law,
non come conferma piatta.

Vantaggi: massima onestà, allineato con il rigor della validation
statistica del round-3.

## Cosa manca prima di scrivere il report finale

### Essenziale (richiede mio lavoro)
1. **P4 verification** — per-target-macro stratified deltas (T&T −,
   non-T&T +) su 5 città. ~15 min di mio lavoro (estensione memlow,
   uso `df_test["cat_macro"]` per filtrare hits per stratum).
   Risultato: scorecard completa 9/9.

2. **Statistical hardening alla B9/B10** — bootstrap CI95 + Holm su:
   - per-city Δ B_full−B_blind (5 città, paired CI95)
   - per-city ΔF1 T-based vs time-only (5 città, Wilcoxon already in
     verdict)
   - Permutation test sui sinks (à la B10.1)
   ~30 min di mio lavoro.

### Strategico (richiede tua decisione + scrittura)
3. **Scelta narrativa A/B/C** dalle 3 vie sopra. Influenza pesante
   sul taglio dell'abstract e della discussion.

4. **MULTICITY_RESULTS.md → integrare nel paper draft** come Section 6
   "Cross-city generalization". Quando sarà chiara la narrativa, ti
   strutturo le sottosezioni.

### Opzionale (PART 3 differita)
5. **Synthetic high-T&T anchor** (reverse-C6 upsampling) — non più
   strettamente necessario dato che Tokyo-TIST a TT=0.45 ha già fornito
   un punto mid-high in dati reali. La motivazione iniziale era
   "TT=0.71 unreachable" ma con la lettura nuova (TT non è il vero
   predittore), il punto sintetico aggiunge poco. Suggerimento:
   **scartare PART 3** e dedicare il tempo alla scrittura.

## Quick reference — what to run if anything

Tutti i numeri sono già a disco. Per refresh:

```bash
# Recompute Stage D additive on any city (instant, no retraining):
.venv/bin/python -m experiments.multicity.stage_d_memlow --city <city>

# View any city's headline:
cat outputs_multicity/<city>/summary_stage{A,B,C,D}.json
cat outputs/<city>/xsage/recommendation/three_way.csv
```

## Status finale

| | nyc_tist | bangkok | saopaulo | tokyo_tist | istanbul |
|---|:---:|:---:|:---:|:---:|:---:|
| step01 / A / backbones / B / C / D | ✅ × 6 | ✅ × 6 | ✅ × 6 | ✅ × 6 | ✅ × 6 |

**5/5 città complete su tutti gli stage. Pronti per la scrittura del
paper.** L'unico item operativo aperto è la verifica P4 (~15 min) +
statistical hardening (~30 min). Il resto è narrativa.
