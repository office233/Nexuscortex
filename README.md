<p align="center">
  <img src="https://img.shields.io/badge/🧠_NEXUS-CORTEX-blueviolet?style=for-the-badge&labelColor=0d1117" height="40" />
</p>

<h3 align="center">A bio-inspired cognitive engine written from scratch in Go</h3>
<h4 align="center">Not an LLM wrapper. Not a transformer clone. A different kind of intelligence.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Go-1.26-00ADD8?style=flat-square&logo=go" />
  <img src="https://img.shields.io/badge/CUDA-Optional-76B900?style=flat-square&logo=nvidia" />
  <img src="https://img.shields.io/badge/Params-500M-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Tests-All_Pass-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p align="center">
  <a href="#-what-is-this">What Is This</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-neural-dashboard">Dashboard</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 🔬 What Is This

Nexus Cortex is a **complete cognitive engine** — modeled after the human brain's architecture — written entirely from scratch in Go with optional CUDA acceleration.

It doesn't call OpenAI. It doesn't wrap Hugging Face. It doesn't use PyTorch.

**It IS the model.**

```
Input → Wernicke (comprehension) → Hippocampus (memory) → Prefrontal (reasoning)
      → Expert Routing → Broca (language generation) → Emotion → Output
```

### Why does this exist?

Every AI project today calls an API. This one builds the brain from first principles:

- **Sparse Distributed Representations** instead of dense embeddings
- **Ternary weights {-1, 0, +1}** instead of float32 — 16x smaller
- **Associative memory** instead of attention-is-all-you-need
- **Sleep consolidation** instead of catastrophic forgetting
- **Curiosity-driven learning** instead of static datasets
- **Emotional modulation** instead of temperature knobs

> Built to explore what happens when you engineer intelligence from biology, not from linear algebra.

---

## 🧬 Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                    NEXUS CORTEX ORGANISM                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌─────────────┐     ┌──────────────┐    ┌──────────────┐   ║
║  │  Wernicke    │────▶│ Hippocampus  │───▶│  Prefrontal  │   ║
║  │ (understand) │     │  (remember)  │    │   (reason)   │   ║
║  └──────┬───────┘     └──────────────┘    └──────┬───────┘   ║
║         │                                        │           ║
║  ┌──────▼───────┐     ┌──────────────┐    ┌──────▼───────┐   ║
║  │   Sensory    │     │   Emotion    │    │    Broca     │   ║
║  │  (encode)    │     │   (modulate) │    │  (generate)  │   ║
║  └──────────────┘     └──────────────┘    └──────────────┘   ║
║                                                              ║
║  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐   ║
║  │  Cerebellum  │     │   Curiosity  │    │    Sleep     │   ║
║  │  (motor)     │     │   (explore)  │    │ (consolidate)│   ║
║  └──────────────┘     └──────────────┘    └──────────────┘   ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │            Fractal Cortex — Expert Routing            │    ║
║  │     ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │    ║
║  │     │ E1 │ │ E2 │ │ E3 │ │ E4 │ │ E5 │ │ En │       │    ║
║  │     └────┘ └────┘ └────┘ └────┘ └────┘ └────┘       │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │     NeuroTexture Compute — RGBA32 Ternary Tiles      │    ║
║  │         CPU / CUDA / WebGPU compute backends         │    ║
║  └──────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════╝
```

### Brain Regions (all implemented)

| Module | Inspired By | What It Does |
|--------|------------|--------------|
| **Wernicke** | Wernicke's area | Language comprehension — encodes input into sparse neural representations |
| **Broca** | Broca's area | Language production — generates output from neural activity |
| **Hippocampus** | Hippocampus | Episodic & semantic memory formation, storage, and retrieval |
| **Prefrontal** | Prefrontal cortex | Reasoning, decision-making, reservoir computing |
| **Cerebellum** | Cerebellum | Motor planning and sequence coordination |
| **Emotion** | Limbic system | Valence-arousal emotional state modulation |
| **Curiosity** | Dopaminergic system | Novelty detection, exploration drive |
| **Sleep** | Sleep cycles | Memory consolidation, synaptic pruning, replay |
| **Sensory** | Sensory cortex | Input encoding and signal processing |
| **Reward** | Reward circuits | Reinforcement learning signals |

### Compute Innovation

| Feature | Traditional LLM | Nexus Cortex |
|---------|:---------------:|:------------:|
| Weight format | float32 (4 bytes) | Ternary {-1,0,+1} (0.25 bytes) |
| Storage per param | 4 bytes | 0.25 bytes (**16x smaller**) |
| Learning | Static (needs retraining) | Continuous (online + sleep) |
| Memory | In weights only | Episodic + Semantic + Working |
| Forgetting | Catastrophic | Controlled (sleep consolidation) |
| Activation | Dense (all params) | Sparse (expert routing) |
| Emotion | None | Valence-arousal vector space |
| GPU requirement | Mandatory | Optional (CPU-first, CUDA optional) |
| External dependencies | PyTorch, CUDA, etc. | **Zero.** Pure Go. |

---

## 🚀 Quick Start

### Prerequisites
- Go 1.21+ (tested on 1.26)
- No other dependencies. Seriously. It's pure Go.

### Build & Run

```bash
# Clone
git clone https://github.com/office233/Nexuscortex.git
cd Nexuscortex

# Build everything (takes ~5 seconds)
go build ./...

# Train on demo corpus (3 epochs, ~30 seconds)
go run ./cmd/cortex-train \
  -data-dir ./data/cortex \
  -corpus ./data/corpus/general.jsonl \
  -epochs 15 \
  -curriculum=true \
  -revisit=true

# Run evaluation
go run ./cmd/cortex-eval -data-dir ./data/cortex

# Start Neural Dashboard
go run ./cmd/cortex-web -port 8080 -data-dir ./data/cortex -open

# Start autonomous learning loop
go run ./cmd/cortex-autonomous -data-dir ./data/cortex
```

### Training Output

```
╔══════════════════════════════════════════════════════════════════╗
║  🧠  NEXUS CORTEX COGNITIVE TRAINER & CURRICULUM SCHEDULER      ║
╚══════════════════════════════════════════════════════════════════╝

[Neurogenesis] Block #0 Active. Total Unique Params: ~500,000,000
📂 Loading training corpus...
📊 Successfully loaded 107 corpus items.
🪜 Applying curriculum learning: sorting items from simple to complex...

🏁 Epoch 1/3 starting...
⏳ Tr:  10/107 | Tok/s: 5617 | Vocab: 56  | Syn: 187
⏳ Tr:  50/107 | Tok/s: 6200 | Vocab: 280 | Syn: 950
⏳ Tr: 107/107 | Tok/s: 6400 | Vocab: 520 | Syn: 1800
💤 Sleep consolidation... replaying 107 episodic memories...
✅ Epoch 1 complete.
```

---

## 🖥️ Neural Dashboard

Real-time visualization of the cognitive engine:

**💬 Cognitive Interaction** — Chat directly with the organism
**📊 Cognitive Vitals** — Synaptic mass, memories, vocabulary, prefrontal neurons
**🔮 Global Workspace** — Prediction error (surprise), attention salience
**🎭 Emotional Compass** — 2D valence-arousal plot with live mood tracking
**⚡ Biological Knobs** — Sleep pressure, alertness, curiosity drive
**💤 Sleep Console** — Watch memory consolidation happen in real-time

```bash
go run ./cmd/cortex-web -port 8080 -data-dir ./data/cortex -open
```

---

## 📊 Key Capabilities

### Implemented & Tested ✅

- **Curriculum learning** — trains from simple to complex
- **Surprise-based replay** — replays high-surprise items more
- **Sleep consolidation** — episodic → semantic memory transfer
- **Beam search decoding** — multiple hypothesis generation
- **Sparse attention** — SDR-based attention mechanism
- **Ternary compute** — RGBA32 packed weights, 16 per uint32
- **CUDA acceleration** — optional GPU kernels for sparse forward pass
- **Fractal architecture** — multiple cortex blocks with expert voting
- **Thousand Brains** — Jeff Hawkins' theory implementation
- **Autonomous learning** — gap detection → Wikipedia search → learn
- **Web learning** — learns from web pages with SSRF hardening
- **Analogy reasoning** — A:B :: C:? style reasoning
- **Fuzz testing** — randomized input resilience
- **CI/CD** — GitHub Actions with `go test` and `go vet`

### Test Results

```
ok   nexus-cortex/cmd/cortex       1.3s    ✅
ok   nexus-cortex/cmd/cortex-web   9.5s    ✅
ok   nexus-cortex/cortex          86.3s    ✅  (30+ tests)
```

---

## 🔬 Research Foundations

This project explores ideas from:

| Theory | Implementation |
|--------|---------------|
| **Sparse Distributed Representations** (Numenta) | `sdr.go`, `sdr_fast.go`, `sdr_pool.go` |
| **Thousand Brains Theory** (Jeff Hawkins) | `thousand_brains.go` |
| **BitNet b1.58** (ternary weights) | `ternary.go`, `neurotexture.go` |
| **Mixture of Experts** (Switch Transformer) | `fractal_cortex.go`, `expert_shard.go` |
| **Global Workspace Theory** (Baars) | `workspace.go` |
| **Predictive Coding** | `predictor.go`, `confidence.go` |
| **Hebbian/STDP Learning** | `error_learning.go`, `reward.go` |
| **Memory Consolidation** (sleep replay) | `sleep_consolidation.go` |
| **Hyperdimensional Computing** | `sdr_attention.go` |

---

## 📁 Project Structure

```
Nexuscortex/
├── cmd/
│   ├── cortex/              # Interactive CLI
│   ├── cortex-train/        # Curriculum trainer
│   ├── cortex-eval/         # Evaluation runner
│   ├── cortex-autonomous/   # Autonomous learning loop
│   ├── cortex-web/          # Neural Dashboard server
│   ├── cortex-tokenizer/    # Tokenizer tools
│   ├── cortex-diagnose/     # System diagnostics
│   ├── corpus-convert/      # Corpus format converter
│   └── train/               # Alternative trainer
├── cortex/                  # Core brain engine
│   ├── brain.go             # Associative neural network
│   ├── organism.go          # Top-level organism wrapper
│   ├── attention.go         # Sparse attention mechanism
│   ├── transformer.go       # Transformer layer
│   ├── hippocampus.go       # Episodic & semantic memory
│   ├── prefrontal.go        # Reasoning & reservoir computing
│   ├── broca.go             # Language generation
│   ├── wernicke.go          # Language comprehension
│   ├── cerebellum.go        # Motor/sequence planning
│   ├── emotion.go           # Valence-arousal system
│   ├── curiosity.go         # Novelty/exploration drive
│   ├── reward.go            # Reinforcement signals
│   ├── sleep_consolidation.go # Memory consolidation
│   ├── fractal_cortex.go    # Multi-block expert system
│   ├── thousand_brains.go   # Thousand Brains Theory
│   ├── quantum_tile.go      # Quantum computing tiles
│   ├── neurotexture.go      # RGBA32 ternary weight format
│   ├── ternary.go           # Ternary weight operations
│   ├── sdr.go               # Sparse Distributed Representations
│   ├── beam_search.go       # Beam search decoding
│   ├── autonomous.go        # Autonomous learning
│   ├── web_learner.go       # Web-based learning
│   ├── compute/             # CPU / CUDA / WebGPU backends
│   └── *_test.go            # 30+ test files
├── cuda/                    # CUDA kernel implementations
├── web/                     # Dashboard (glassmorphism UI)
├── data/
│   ├── corpus/              # Training corpora
│   └── evals/               # Evaluation suites
├── docs/                    # Research docs & benchmarks
└── .github/workflows/       # CI/CD
```

---

## 🛠️ Tech Stack

| Layer | What |
|-------|------|
| **Language** | Go 1.21+ — zero Python, zero JS frameworks |
| **Compute** | CPU-first, optional CUDA kernels |
| **Weight format** | RGBA32 ternary tiles (0.25 bytes/param) |
| **Storage** | JSON persistence + planned NTX binary format |
| **Dashboard** | Vanilla HTML/CSS/JS with glassmorphism |
| **CI** | GitHub Actions (`go test`, `go vet`) |
| **Dependencies** | **None.** Pure Go standard library. |

---

## 🗺️ Roadmap

- [x] Core brain with 10 neural regions
- [x] Curriculum training with surprise-based replay
- [x] Sleep consolidation
- [x] Neural Dashboard with emotional compass
- [x] Autonomous learning loop
- [x] CUDA compute backend
- [x] 30+ unit tests, all passing
- [x] CI/CD pipeline
- [ ] NTX binary checkpoint format (mmap-friendly)
- [ ] Expert Atlas with disk-backed experts
- [ ] Top-K expert routing (replace all-block voting)
- [ ] Broca 2.0 autoregressive generator (50-150M params)
- [ ] BPE tokenizer (32K vocab, Romanian + English)
- [ ] Benchmark arena (1000+ test cases)
- [ ] WebGPU compute backend

---

## 🤔 FAQ

**Is this a real AI?**
It's a real cognitive engine that learns, remembers, reasons, and generates language. It's not GPT — it's a fundamentally different architecture inspired by neuroscience.

**Can it replace ChatGPT?**
No. It's a research prototype exploring non-transformer approaches to intelligence. Its strength is in continuous learning, memory, and sparse compute — not in raw language fluency.

**Why Go?**
Speed, simplicity, easy concurrency, single binary output, no dependency hell. Go compiles the entire project in 5 seconds.

**Do I need a GPU?**
No. CPU-first design. CUDA is optional and only accelerates sparse ternary forward passes.

**How many parameters?**
~500M with a single cortex block. Scales with FractalCortex blocks.

---

## 📝 License

MIT — Use it, fork it, build on it.

---

<p align="center">
  <strong>⭐ Star this if you believe intelligence can be built differently.</strong>
</p>

<p align="center">
  Built from scratch. No frameworks. No wrappers. No shortcuts.
</p>
