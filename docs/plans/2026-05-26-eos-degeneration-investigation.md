# Investigation: EOS Degeneration in Cursa D

**Date**: 2026-05-26
**Context**: Postmortem cursa D (vezi `2026-05-25-cursa-D-postmortem.md`)
observed that the average factual generation length collapsed from
21.1 chars (step 10500) to 0.9 chars (step 30000), with 75%+ empty
generations. This document records the root-cause investigation and
its implications for cursa E.

## Hypothesis A (rejected): `EOSTokenID = 3 → "photosynthesis"`

Initial suspicion was that `Config.TransformerEOSTokenID = 3` mapped
to an ordinary corpus word ("photosynthesis"), causing premature
termination. **Rejected after inspection** of
`data/cortex-auto/tokenizer.json`:

- The BPE tokenizer correctly reserves IDs 0-4 for special tokens:
  `<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3, <SEP>=4`
- The other file `data/cortex-auto/vocab.json` is unrelated — it is
  used by the hippocampus/brain memory subsystem, not the transformer
- `EOSTokenID = 3` is correctly the `<EOS>` special token

Lesson: there are **two parallel vocabularies** in the data dir, with
overlapping IDs but completely separate purposes. The transformer
exclusively uses `tokenizer.json` (BPE).

## Hypothesis B (rejected): training degradation over time

Initial reading of `eval-history.jsonl` suggested gen length was
collapsing as a function of training step. **Rejected after detailed
per-step analysis** (`scripts/analyze-gen-length` and
`scripts/check-eos-rate`):

- Empty-generation rate is **already ~75% at step 10500** (the
  earliest available eval), and stays at 75-80% throughout
- The "degradation" is mostly noise: step-to-step variance is huge
  (overall avg ranges from 0.6 to 31.1 chars across consecutive evals)
- Math category actually trended **upward** (+166% in last 5 vs
  first 5 evals), inconsistent with a global degradation story

The "step 30000 has 0.6 chars avg" was an unlucky outlier, not the
endpoint of a trend.

## Hypothesis C (confirmed): training corpus is dominated by short sequences

`scripts/corpus-length-dist` measured the word-count distribution of
training data:

| Bucket    | Count   | Pct    |
|-----------|---------|--------|
| <10 words | 39064   | 15.5%  |
| 10-25     | 132176  | **52.3%** |
| 25-50     | 44925   | 17.8%  |
| 50-100    | 22439   | 8.9%   |
| 100-200   | 10666   | 4.2%   |
| 200-500   | 2970    | 1.2%   |
| ≥500      | 480     | 0.2%   |

**67.8% of training sequences are under 25 words** (≈30-35 BPE tokens
including specials). The median sequence is 19 words.

Combined with the fact that every training sequence ends with `<EOS>`
(via `EncodeWithSpecial`), the model learns a strong prior:
`P(<EOS> | ~20-30 tokens generated) = HIGH`.

At eval time, prompts are ~7-10 tokens. After the model emits ~10-20
tokens of "response", `<EOS>` becomes the most likely next token
regardless of whether the response makes sense.

## Empirical confirmation via DUAL eval

Implemented `GenerateFastMin(prompt, maxNew, minNewTokens, temp, topK)`
that suppresses the EOS logit for the first `minNewTokens` emitted
tokens (set to -Inf before top-K). Added `--min-tokens` flag to
`broca-eval`. Tests:

- `TestGenerateFastMinEquivalentToGenerateFast`: `min=0` is pure
  pass-through (no behavioural drift on existing call sites)
- `TestGenerateFastMinSuppressesEOS`: with `min=N`, EOS never appears
  in the first N emitted tokens

Ran the standard 24-task eval suite on `transformer.best.nxtf` twice:

| Run | min-tokens | Empty gens | Avg gen length |
|-----|------------|------------|----------------|
| A   | 0 (baseline) | 18/24 (75%) | 24.2 chars |
| B   | 15           | 0/24 (0%)   | 88.8 chars |

**Both runs scored 0/24 correct**. The longer generations from run B
were syntactically plausible but semantically irrelevant. Examples:

- `"What is the capital of Germany?"` → `"The Sleep is the most
  important country in the United States..."` (vs expected: berlin)
- `"What is 6 × 14?"` → `"5, 5 minutes is equal to 1. 6. 2. 15. 4.
  2. 4. 2."` (vs expected: 84)
- `"What is the chemical formula for water?"` → `"The capital of
  France is the United States, Germany, France..."` (vs expected: h2o)

## Verdict

**The 5.4M-param model does not know the factual answers.** EOS
suppression does not surface hidden knowledge — it merely forces the
model to fabricate plausible-sounding text. The empty generations in
baseline were the model "honestly refusing" to answer (high P(EOS)
right after the question) rather than hiding correct answers behind
an early EOS.

## Implications for cursa E

1. **Do NOT use `--min-tokens` for cursa E evals.** It does not
   improve scoring and contaminates comparability with cursa C/D
   history.

2. **Do NOT reformat the training corpus yet.** A larger model
   (cursa E: 15M params) may have enough capacity to learn that
   factual questions require specific answers — without any data
   reformatting. If cursa E still scores 0% factual, then we revisit
   the data pipeline (instruction-tuning format, longer sequences,
   answer-prefix conventions).

3. **Track gen length per category in cursa E eval history** as a
   diagnostic signal, but do NOT optimize for it.

4. **The `--min-tokens` flag and `GenerateFastMin` API are kept** as
   permanent additions. Useful for future debugging ("does the model
   know it but stops?") and for any task where minimum response
   length is a hard requirement.

## Code changes shipped with this investigation

- `cortex/transformer_cache.go`: new `GenerateFastMin`. `GenerateFast`
  is now a thin wrapper with `minNewTokens=0`.
- `cortex/cot.go`: `CoTConfig.MinTokens` field; routes through
  `GenerateFastMin`.
- `cmd/broca-eval/main.go`: `--min-tokens N` flag, persisted in the
  JSON report as `min_tokens`.
- `cortex/transformer_cache_test.go`: 2 new tests
  (`TestGenerateFastMinEquivalentToGenerateFast`,
  `TestGenerateFastMinSuppressesEOS`).
- `scripts/analyze-gen-length/`, `scripts/check-eos-rate/`,
  `scripts/corpus-length-dist/`, `scripts/inspect-tokenizer/`,
  `scripts/compare-dual/`: diagnostic tools.
