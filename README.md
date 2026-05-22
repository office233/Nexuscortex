# 🧠 Nexus Cortex

**A bio-inspired, self-learning AI engine built from scratch in Go.**

> ⚠️ **This is an experimental research prototype**, not a production-ready language model. It explores alternative neural architectures inspired by neuroscience — spiking neural networks, sparse distributed representations, and Hebbian learning — to investigate whether non-transformer approaches can achieve meaningful language understanding.

[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go)](https://go.dev)
[![Tests](https://img.shields.io/badge/tests-23_files_passing-brightgreen?style=flat)]()
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat)]()
[![Lines of Code](https://img.shields.io/badge/code-16%2C000+_LoC-orange?style=flat)]()

---

## What Is This?

Nexus Cortex is a **from-scratch neural architecture** that does NOT use transformers, attention heads, or backpropagation. Instead, it draws from computational neuroscience:

| Component | Inspiration | What It Does |
|-----------|-------------|-------------|
| **Spiking Neurons** | Biological neural firing | Event-driven processing, not continuous activation |
| **SDR Encoding** | Numenta's HTM theory | Sparse Distributed Representations for semantic similarity |
| **Ternary Weights** | BitNet b1.58 | Weights are {-1, 0, +1} — zero multiplications, only ADD/SUB |
| **Hebbian/STDP Learning** | Synaptic plasticity | "Neurons that fire together, wire together" |
| **Hippocampal Memory** | Episodic memory | Pattern-completion retrieval via SDR similarity |
| **ThousandBrains** | Numenta's cortical columns | Multiple independent predictions, majority voting |
| **Linear Scan** | RWKV / State Space Models | O(1) memory per token, no KV cache |
| **Autonomous Learning** | Curiosity-driven exploration | Self-directed web learning from Wikipedia + HuggingFace |

### The Key Idea

Most LLMs are frozen after training — they never learn anything new. Nexus Cortex **learns continuously**: from conversations, from training data, and autonomously from the internet. Every interaction makes it slightly smarter. This is its core differentiator, not raw benchmark performance.

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         INPUT TEXT               │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │    Wernicke (Language Parser)     │
                    │    Tokenize → N-gram context      │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │    Encoder (SDR Encoder)          │
                    │    Text → Sparse Binary Vector    │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼──────┐ ┌───────▼──────┐
    │  Cerebellum     │ │ Hippocampus │ │    Brain     │
    │  (Fast Cache)   │ │ (Episodic   │ │ (Associative │
    │  O(1) lookup    │ │  Memory)    │ │  Network)    │
    │  System 1       │ │ SDR recall  │ │ Word links   │
    └─────────┬──────┘ └──────┬──────┘ └───────┬──────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │   Prefrontal (Executive Control)  │
                    │   Multi-hop reasoning + planning  │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │   Broca (Language Production)     │
                    │   SDR → Text generation           │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │         OUTPUT TEXT               │
                    └─────────────────────────────────┘
```

### Memory Hierarchy (4 Tiers)

| Tier | Module | Speed | Capacity | Persistence |
|------|--------|-------|----------|-------------|
| L1 | **Cerebellum** (cache) | O(1) | ~1K entries | Session |
| L2 | **Hippocampus** (episodic) | O(N) SDR scan | ~10K memories | Saved |
| L3 | **Brain** (associative) | O(1) hash | ~3M synapses | Saved |
| L4 | **SemanticMemory** (concepts) | O(N) | ~1K concepts | Saved |

---

## Quick Start

### Prerequisites
- **Go 1.21+** ([install](https://go.dev/dl/))
- No GPU required (runs on CPU)
- ~1-2 GB RAM

### Build
```bash
git clone https://github.com/office233/Nexuscortex.git
cd Nexuscortex
go build ./...
```

### Run Tests
```bash
go test ./cortex/... -count=1 -v
```

### Interactive Chat
```bash
go run ./cmd/cortex
```

### Train on Corpus
```bash
# Download training data first (see Data section below)
go run ./cmd/cortex-train -corpus "./data/corpus/alpaca.jsonl" -epochs 3
```

### Autonomous Self-Learning
```bash
# Starts learning from Wikipedia + HuggingFace automatically
go run ./cmd/cortex-autonomous -interval 15 -gaps 3
```

### Web Interface
```bash
go run ./cmd/cortex-web
# Open http://localhost:8080
```

---

## Training Data

The training corpus is NOT included in this repo (too large). Download it:

```bash
# Option 1: Use the download script
python scripts/download_hf_corpus.py

# Option 2: Manual download
# Place .jsonl files in data/corpus/
```

**Included** (small, for testing):
- `data/corpus/general.jsonl` — 107 basic Romanian examples
- `data/corpus/reasoning.jsonl` — 43 step-by-step reasoning examples (EN + RO)

**Recommended downloads:**
| Dataset | Size | Content |
|---------|------|---------|
| [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 36 MB | 52K instruction-response pairs |
| [Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 18 MB | 15K diverse Q&A |
| [GSM8K](https://github.com/openai/grade-school-math) | 4 MB | 7.5K math reasoning chains |

---

## Benchmarks (Honest Numbers)

Measured on a single machine: AMD Ryzen / Intel i7, GTX 1660 Ti 6GB, 16GB RAM.

### Training Performance
| Metric | Value |
|--------|-------|
| Training speed | ~1,600-2,400 tok/s |
| Vocabulary after Alpaca (191K items) | 53,058 words |
| Synapses after training | 2,793,000 |
| Training time (Alpaca, 1 epoch) | ~2 minutes |
| RAM usage during training | ~1.2 GB |

### Inference Performance
| Pipeline | Tokens/sec | Notes |
|----------|-----------|-------|
| Fast path (Cerebellum cache hit) | ~53,000 | Cached responses |
| Full pipeline (Hippocampus + Brain) | ~130-200 | Novel queries |
| CortexStack (ternary deep layers) | ~137 | 12-layer, 1024-dim |

### Language Quality (Honest Assessment)
| Test | Score | Notes |
|------|-------|-------|
| Seen Training Recall | 7/7 | Memorized training examples |
| Unseen Generalization | 3/12 | **Weak** — this is where work is needed |
| Self-test (autonomous learner) | 0-12.5% | Early stage, improving |

> **Note:** This system does NOT compete with GPT-4, Claude, or other large language models on general benchmarks. It is a research prototype exploring alternative architectures. Its strengths are in continuous learning, extreme efficiency, and architectural novelty — not in raw language quality.

---

## Key Technical Features

### Ternary Engine (`cortex/ternary.go`)
Weights are restricted to {-1, 0, +1}, packed as 2 bits each (16 weights per uint32). Forward pass uses only ADD and SUB — **zero floating-point multiplications**. This is architecturally similar to Microsoft's [BitNet b1.58](https://arxiv.org/abs/2402.17764).

### SDR Attention (`cortex/sdr_attention.go`)
Sparse Distributed Representations enable semantic similarity via Hamming distance. A 2048-bit SDR with ~40 active bits provides collision-resistant encoding with O(1) comparison.

### Linear Scan (`cortex/linear_scan.go`)
Inspired by RWKV and state-space models, processes sequences with O(1) memory per step — no KV cache, constant memory regardless of sequence length.

### Autonomous Learning (`cortex/autonomous.go`)
The self-learning loop:
1. **Curiosity** — Identifies knowledge gaps from low-confidence responses
2. **Search** — Queries Wikipedia API and HuggingFace datasets
3. **Learn** — Feeds discoveries through Brain/Hippocampus/Wernicke
4. **Evaluate** — Self-generates tests and measures improvement
5. **Consolidate** — Periodic sleep cycles for memory organization
6. **Repeat** — Forever

### ALBERT Weight Sharing (`cortex/ternary.go: SharedCortexStack`)
Reuses a single layer's weights across N virtual layers (like [ALBERT](https://arxiv.org/abs/1909.11942)), enabling deep computation with minimal memory.

---

## Project Structure

```
nexus-cortex/
├── cmd/
│   ├── cortex/              # Interactive CLI
│   ├── cortex-train/        # Corpus training
│   ├── cortex-eval/         # Evaluation harness
│   ├── cortex-autonomous/   # Self-learning engine
│   ├── cortex-web/          # Web interface
│   └── corpus-convert/      # Dataset format converter
├── cortex/                  # Core engine (74 files, 16K+ LoC)
│   ├── organism.go          # Main orchestrator
│   ├── brain.go             # Associative word network
│   ├── hippocampus.go       # Episodic memory (SDR)
│   ├── encoder.go           # Text → SDR encoder
│   ├── wernicke.go          # Language parser (n-grams)
│   ├── broca.go             # Language generator
│   ├── prefrontal.go        # Executive reasoning
│   ├── cerebellum.go        # Fast response cache
│   ├── ternary.go           # Ternary {-1,0,+1} compute engine
│   ├── linear_scan.go       # O(1) sequence processing
│   ├── sdr.go               # Sparse Distributed Representations
│   ├── autonomous.go        # Self-learning loop
│   ├── web_learner.go       # Wikipedia/HuggingFace fetcher
│   ├── self_evaluator.go    # Auto-testing system
│   ├── thousand_brains.go   # Cortical column voting
│   └── *_test.go            # 23 test files
├── data/
│   ├── corpus/              # Training data (.jsonl)
│   ├── evals/               # Evaluation test sets
│   └── demo/                # Demo configuration
├── web/                     # Web UI (HTML/CSS/JS)
├── docs/                    # Documentation & plans
└── scripts/                 # Utility scripts
```

---

## Limitations

This is a research prototype. Here's what it **cannot** do well (yet):

- ❌ **General language quality** — Responses are often incoherent or repetitive for novel queries
- ❌ **Complex reasoning** — No chain-of-thought; reasoning is shallow pattern matching
- ❌ **Long-form generation** — Limited coherence beyond 1-2 sentences
- ❌ **Factual accuracy** — No grounding; may hallucinate or confuse learned facts
- ❌ **Scale** — 2.8M synapses vs billions of parameters in production LLMs
- ❌ **No GPU acceleration** — Ternary engine runs on CPU only (GPU compute shaders planned)

### What It CAN Do

- ✅ **Learn continuously** — Every interaction strengthens synaptic connections
- ✅ **Learn autonomously** — Searches Wikipedia/HuggingFace without human intervention
- ✅ **Run locally** — No cloud, no API keys, complete privacy
- ✅ **Extreme efficiency** — 1.2 GB RAM, zero multiplications, runs on any hardware
- ✅ **Self-evaluate** — Generates tests and tracks improvement over time

---

## Research Inspirations

| Paper/System | How It Influenced This Project |
|-------------|-------------------------------|
| [Numenta HTM](https://numenta.com/resources/research-publications/) | SDR encoding, Thousand Brains Theory |
| [BitNet b1.58](https://arxiv.org/abs/2402.17764) | Ternary {-1,0,+1} weight validation |
| [RWKV](https://github.com/BlinkDL/RWKV-LM) | Linear Scan O(1) state design |
| [ALBERT](https://arxiv.org/abs/1909.11942) | Weight sharing across layers |
| [Phi-4](https://arxiv.org/abs/2306.11644) | Data quality > model size philosophy |
| [DeepSeek R1](https://arxiv.org/abs/2401.02954) | RL-based reasoning emergence |
| [MatMul-Free LM](https://arxiv.org/abs/2406.02528) | Multiplication-free architecture validation |

---

## Roadmap

- [x] Core SNN architecture (spiking neurons, SDR, Hebbian learning)
- [x] 4-tier memory hierarchy (Cerebellum → Hippocampus → Brain → Semantic)
- [x] Ternary compute engine (zero multiplications)
- [x] Curriculum training with spaced repetition
- [x] Autonomous self-learning from Wikipedia
- [x] Self-evaluation and progress tracking
- [ ] HuggingFace dataset integration (in progress)
- [ ] GPU compute shaders for ternary forward pass
- [ ] Speculative decoding (ternary draft model)
- [ ] Reward-modulated learning (GRPO-inspired)
- [ ] Process Reward Model for step verification
- [ ] Web UI improvements (live training dashboard)

---

## Contributing

This is a personal research project, but contributions and discussions are welcome! If you're interested in:
- Alternative neural architectures
- Bio-inspired computing
- Efficient AI on consumer hardware
- Continuous/lifelong learning

Feel free to open an issue or PR.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built with curiosity and Go. No transformers were harmed in the making of this project.</em>
</p>
