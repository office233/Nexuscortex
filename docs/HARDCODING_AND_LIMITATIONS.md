# Hardcoding, Heuristics & Limitations

> **Purpose**: This document declares all operational defaults, heuristic
> thresholds, seed data, and known limitations in the Nexus Cortex codebase.
> Nothing here is hidden — this is the honest inventory.

## 1. Seed Data (Configurable via Config)

All seed data is now stored in `Config` struct fields (see `config.go`) and
can be overridden via JSON config file or programmatic construction.

### Seed Topics (`AutoSeedTopics`)
Default: 32 topics across science, math, technology, history, geography,
logic, and Romanian culture. These bootstrap the AutonomousLearner's
curiosity. They are **not** training data — they are search queries for
Wikipedia/HuggingFace.

### Seed Datasets (`AutoSeedDatasets`)
Default: `tatsu-lab/alpaca`, `gsm8k`, `hellaswag`. These are well-known
public instruction/reasoning datasets used for initial learning. They can be
changed or emptied via config.

### Search Languages (`AutoSearchLangs`)
Default: `["en", "ro"]`. Determines which Wikipedia languages are searched.

## 2. Eval Data

`data/evals/comprehensive.jsonl` contains ~30 test cases across 4 categories:
recall, generalization, wikipedia, reasoning. This is a **small** eval set
suitable for development iteration, not for capability claims. Expected
answers are fixed strings and partial-match scoring is used.

**Limitation**: This eval set is too small to make general intelligence
claims. It tests basic retrieval and symbolic reasoning only.

## 3. Reasoning Engine (`reasoning.go`)

The reasoning engine uses **symbolic/rule-based** pattern matching:
- Arithmetic: regex-based expression parser
- Sequences: detects arithmetic and geometric progressions
- Logic: parses simple syllogisms ("All X are Y. Z is X. Is Z Y?")
- Sorting: parses and sorts numeric lists

**This is NOT neural reasoning.** It is a deterministic symbolic shim that
provides correct answers for structured math/logic queries. It should not be
presented as evidence of emergent reasoning ability.

## 4. Operational Defaults

| Setting | Default | Source |
|---------|---------|--------|
| Dashboard port | `8080` | `Config.WebPort` |
| Bind address | `127.0.0.1` | `Config.WebBindAddr` |
| RNG seed | `42` | `Config.Seed` |
| SDR size | `10000` | `Config.SDRSize` |
| Active bits | `50` | `Config.ActiveCount` |
| Max memories | `10000` | `Config.MaxMemories` |
| Learn interval | `30s` | `Config.AutoLearnInterval` |
| HTTP timeout | `10s` | hardcoded in `NewWebLearner()` |
| Rate limit | `2s` | hardcoded in `NewWebLearner()` |
| HTTP body limit | `5 MB` | `io.LimitReader` in all HTTP reads |

All `Config` fields can be overridden. WebLearner timeout/rate-limit are
currently hardcoded but are reasonable for polite web scraping.

## 5. File Permissions

All model/state files use `0600` (owner read/write only).
All model/state directories use `0700` (owner read/write/execute only).
This is applied uniformly across all persistence code.

## 6. External API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `en.wikipedia.org/api/rest_v1/` | Wikipedia summaries |
| `en.wikipedia.org/w/api.php` | Wikipedia search |
| `huggingface.co/api/datasets` | HuggingFace dataset search |
| `datasets-server.huggingface.co/rows` | HuggingFace row download |

All HTTP responses are limited to 5 MB via `io.LimitReader`.
No API keys are required or stored.

## 7. Python Script (`scripts/download_hf_corpus.py`)

- Uses `trust_remote_code=False` for all `load_dataset` calls
- Pinned to specific Wikipedia snapshots (`20231101.ro`, `20231101.en`)
- Article limits are hardcoded (configurable via CLI args would be better)

## 8. Known Limitations

1. **Not an LLM**: Nexus Cortex does not generate fluent text. It retrieves
   and recombines learned fragments.
2. **Small eval set**: 30 test cases is insufficient for capability claims.
3. **Symbolic reasoning only**: Math/logic is rule-based, not learned.
4. **No adversarial testing**: No fuzz tests for corrupt model files yet.
5. **WebGPU untested in CI**: Build tag `webgpu` is not tested in CI due to
   GPU driver requirements.
6. **Single-machine only**: No distributed training or inference.

## 9. What Is NOT Hardcoded

- No API keys, tokens, passwords, or secrets in the codebase.
- No absolute hardcoded production paths; relative defaults exist for local
  development (`./data/cortex` in Config, `data/corpus` in scripts).
- No hardcoded model weights (all learned or randomly initialized).
