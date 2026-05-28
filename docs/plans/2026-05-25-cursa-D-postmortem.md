# Cursa D — Postmortem

**Data**: 25 Mai 2026
**Durată**: ~8 ore wall-time (18:23 → ~02:30)
**Configurație**: continuare din cursa C (resume Adam state @ step 20200)
**Verdict**: ✓ training-ul a funcționat tehnic, ✗ modelul a rămas incapabil pe eval suite

## Ce voiam să aflăm

Cursa C terminase cu val_loss 5.32 și accuracy 0/24. Întrebarea pentru D:
**modelul mai poate învăța dacă îi mai dăm 20.000 de pași cu LR scăzut?**

Configurația de bază a păstrat tot ce mergea în C și a schimbat doar:

| Parametru | Cursa C | Cursa D | De ce |
|---|---|---|---|
| `peak_lr` | 2e-4 | **5e-5** | LR de fine-tuning, nu de pre-training |
| `warmup` | 2000 | **600** | scalat pentru sesiunea mai scurtă |
| `total_steps` | 10000 | **30000** | +20000 steps suplimentari |
| `early_stop` | (lipsea) | **8 evals** | protecție contra waste |
| `auto_eval` | NU | **CoT × 5 samples** | măsurăm la fiecare new-best |

## Ce s-a întâmplat

### Training-ul în sine — SUCCES tehnic

- val_loss a scăzut consistent: **5.32 → 5.02** (-5.6%)
- perplexity: **204 → 151** (-26%)
- Niciun divergence, niciun NaN, niciun memory leak
- Auto-eval hook a funcționat impecabil: 38 puncte în `eval-history.jsonl`
- Throughput stabil ~520 tok/s pe GPU (după ce am rezolvat CUDA DLL discovery)

### Calitatea modelului — EȘEC complet

Accuracy raportată: 0/24 → 1/24 → 0/24 (cu două spike-uri trecătoare la step 12000 și 19500).

**Investigația spike-urilor a relevat că sunt false-positive de scoring**:

#### Spike 1 — step 12000, `reason_syllogism`
- **Expected**: `yes OR no` (ModeContainsAny)
- **Generated**: `"In your mindful of buno can help people know from the city."`
- **De ce s-a marcat corect**: substring `"no"` apare în `"k**no**w"`
- **Realitate**: răspuns gibberish, fără legătură cu pinguini, păsări sau zbor

#### Spike 2 — step 19500, `math_sqrt`
- **Expected**: `9` (ModeNumeric)
- **Generated**: `"9. In the average of 1513 square- 24 million for 94. In the two three people..."`
- **De ce s-a marcat corect**: regex `\d+` extrage primul număr = `9`
- **Realitate**: răspunsul începe accidental cu "9.", restul e total fără legătură

### Investigația factuală — confirmă incapacitatea

Am verificat 38 evaluări × 6 task-uri factuale (capital France/Japan/Germany, Romeo+Juliet,
relativity, largest planet) = **228 încercări**. Modelul **niciodată** nu a generat
"paris", "tokyo", "berlin", "shakespeare", "einstein", sau "jupiter".

La step 30000 (final):
- 7/8 răspunsuri factuale sunt **vide** (empty string)
- generation length medie: **0.9 caractere**
- singura non-empty: "Ye Date" la Romeo & Juliet

### Pattern îngrijorător: degenerarea generației

Lungimea medie a generațiilor factuale a scăzut în timp:

| Step | Avg gen length (factual) | Empty / 8 |
|---|---|---|
| 10500 | 21.1 | 7/8 |
| 13500 | 41.2 | 5/8 |
| 18000 | 25.6 | 5/8 |
| 23500 | 38.1 | 5/8 |
| 27000 | 5.8 | 6/8 |
| **30000** | **0.9** | **7/8** |

Modelul învață să **producă mai puțin** în timp. Suspiciune: probabilitatea tokenului
EOS / punct devine dominantă pe măsură ce modelul converge pe distribuția training.

## Ce am învățat sigur

### 1. Corpus-ul NU este blocaj
Tool nou: `cmd/corpus-eval-coverage` (creat în această sesiune) a confirmat că pe
256.928 linii din dolly+alpaca:
- 22/24 task-uri sunt GREEN (≥10 linii cu co-occurence prompt-context + answer)
- 2/24 sunt YELLOW (1-9 linii)
- **0/24 sunt RED**

Exemplu: `fact_capital_france` are 196 de linii cu "paris" + "france" împreună
("Embassy in Paris, France from 1966...", "Parc de Saint-Cloud, Paris, France...").
Informația **există** în corpus.

### 2. Numărul de steps NU este blocaj
val_loss a scăzut continuu până la step 30000. Modelul ar continua să se
îmbunătățească dacă l-am lăsa, dar trend-ul indică o asimptotă în jurul val_loss 4.8,
care e încă mult prea departe de un model coherent.

### 3. Capacitatea modelului ESTE blocaj
- **5.4M parametri** la vocab 8192 = ~660 params/token
- **Perplexity 151** = top-1 token are probabilitate medie ~0.7%
- **Zero generație factuală corectă** în 228 încercări = nu e o problemă de sampling, e
  o problemă de model

### 4. Bug-uri în `evalsuite/score.go` (de reparat separat)
- `ModeContainsAny` cu `strings.Contains` produce false-positive (substring fără word boundary)
- `ModeNumeric` cu `reNumber.FindString` extrage primul număr indiferent de context

**Decizie**: NU repar acum. Motivație: schimbarea scorerului ar face cursa C/D necomparabile
cu cursele viitoare. Repar în sesiune dedicată, înainte de cursa F sau mai târziu, când
restartăm baseline-ul. Documentat în HARDCODING_AND_LIMITATIONS.md ca TODO.

## Propunere — Cursa E

**Singura variabilă schimbată**: **scale modelul**. Toate celelalte rămân identice.

| Aspect | Cursa D | Cursa E |
|---|---|---|
| params | 5.4M | **~15M** (3×) |
| embed_dim | 256 | **384** |
| layers | 4 | **6** |
| heads | 4 | **6** |
| FFN dim | 1024 | **1536** |
| vocab | 8192 | 8192 (nemodificat) |
| seq_len | 512 | 512 (nemodificat) |
| corpus | dolly + alpaca | dolly + alpaca (același) |
| total_steps | 30000 | **40000** (fresh start, fără resume) |
| peak_lr | 5e-5 | **3e-4** (pre-training rate, model nou) |
| warmup | 600 | **2000** |
| auto_eval | da, CoT × 5 | da, CoT × 5 |

**VRAM estimat**: ~3 GB (vs 1.4 GB la D). Fezabil pe GTX 1660 Ti (6 GB).

**Wall-time estimat**: ~12-15 ore (3× steps × ~1.5× cost per step). Noapte întreagă.

**Risc**: dacă cursa E rezultă tot accuracy 0%, atunci ipoteza "model prea mic" e
falsificată și problema e altundeva (corpus quality, scoring, sau ceva mai subtil în
arhitectură). În acel caz, următoarea direcție ar fi 30M params SAU schimbarea
corpus-ului spre QA-uri mai directe (TriviaQA, NaturalQuestions snippets).

**Risc invers**: dacă cursa E sare la accuracy 20%+ deodată, asta confirmă că scala era
tot ce lipsea — și putem investi în 30M+ pentru cursa F.

## Status acum

- Cursa D: TERMINATĂ, model salvat în `data/cortex-auto/transformer.best.nxtf`
- Cursa E: **NEAPĂRAT confirmare umană** înainte de lansare (e noapte întreagă)
- Tool nou `corpus-eval-coverage` disponibil pentru analize viitoare
- Postmortem documentat (acest fișier)

## TODO ridicate de această cursă

1. **Repară bug-urile evalsuite** (ModeContainsAny word-boundary, ModeNumeric position-aware)
   — sesiune dedicată, înainte de baseline reset
2. **Investighează degenerarea generation length** — de ce scade lungimea medie în timp?
   E ceva în training loop care favorizează EOS prematur? Sau e fenomen normal de
   convergență pe distribuția unui corpus cu răspunsuri scurte?
3. **Build script automat pentru CUDA DLL** (deja TODO din §10 HARDCODING.md) —
   ar fi evitat 10 min de debugging la începutul cursei D
