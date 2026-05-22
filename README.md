# Nexus Cortex

Experimental bio-inspired learning system written in Go.

Nexus Cortex explores non-transformer approaches to learning, memory, evaluation, and autonomous knowledge acquisition. The project is a research/prototype system rather than a production LLM. It is designed to experiment with sparse representations, curriculum learning, memory consolidation, and evaluation loops in a small local runtime.

## What this project demonstrates

- **Bio-inspired architecture experiments**: sparse representations, memory tiers, Hebbian/STDP-inspired learning ideas, and consolidation cycles.
- **Go systems engineering**: command-line tools, training loops, evaluation runners, web dashboard, persistent organism state, and corpus processing utilities.
- **Learning and evaluation workflow**: corpus conversion, training, autonomous learning loop, eval suites, and capability reporting.
- **Local-first AI experimentation**: designed to run on consumer hardware without relying on hosted LLM APIs.
- **Research discipline**: explicit evaluation, confidence reporting, failure cases, and test-suite based scoring.

## Important framing

This is an **experimental research prototype**. It is not presented as a replacement for frontier LLMs, nor as a production-grade model training stack.

The goal is to explore ideas such as:

- sparse distributed representations;
- lightweight memory systems;
- curriculum learning;
- surprise-based replay;
- autonomous learning loops;
- memory consolidation;
- interpretable evaluation harnesses.

For portfolio purposes, the value of this project is the systems design, experimentation discipline, and ability to build a working AI research prototype from scratch.

## Architecture overview

```text
Nexuscortex/
├── cmd/
│   ├── corpus-convert/     # Converts mixed JSONL corpora into canonical training data
│   ├── cortex-autonomous/  # Runs continuous autonomous learning cycles
│   ├── cortex-eval/        # Runs comprehensive evaluation suites and capability reports
│   ├── cortex-train/       # Curriculum training and surprise-based replay
│   ├── cortex-web/         # Local Web Neural Dashboard / Introspection Server
│   └── cortex-diagnose/    # System configuration diagnostic utility
├── cortex/                    # Core organism, encoder, memory, prediction, eval, and learning logic
├── web/                       # Dashboard assets / web server helpers
├── data/
│   ├── corpus/                # Small demo corpora; large corpora are gitignored
│   ├── evals/                 # Evaluation suites (including hidden_benchmark.jsonl)
│   └── cortex/                # Runtime state; generated artifacts are gitignored
└── .gitignore                 # Excludes models, caches, binaries, and generated state
```

## Main workflows

### 1. Convert corpus data

```bash
go run ./cmd/corpus-convert input.jsonl data/corpus/general.jsonl
```

Supported input formats include instruction/response, prompt/completion, GSM8K-style question/answer, and plain text records.

### 2. Train a local organism

```bash
go run ./cmd/cortex-train \
  -data-dir ./data/cortex \
  -corpus ./data/corpus/general.jsonl \
  -epochs 15 \
  -curriculum=true \
  -revisit=true
```

The trainer supports deterministic seeds, curriculum sorting, surprise-based revisit batches, sleep/consolidation after epochs, and persistent organism state.

### 3. Run evaluation suites

```bash
# Run standard evaluation on the comprehensive suite
go run ./cmd/cortex-eval -data-dir ./data/cortex

# Run the strict Romanian + English hidden benchmark suite
go run ./cmd/cortex-eval -data-dir ./data/cortex -eval ./data/evals/hidden_benchmark.jsonl
```

The eval runner reports failed test cases, category scoring breakdown, word-overlap performance metrics, and a total composite capability score.

### 4. Run autonomous learning loop

```bash
go run ./cmd/cortex-autonomous -data-dir ./data/cortex
```

This loop is designed for continuous autonomous learning cycles utilizing gap detection, Wikipedia search, and HuggingFace training segments.

### 5. Start local dashboard

```bash
go run ./cmd/cortex-web -port 8080 -data-dir ./data/cortex -open
```

The dashboard provides a premium real-time visualization of cognitive drives, prefrontal synaptic weights, emotional valence vector states, and conversational introspection.

## Evaluation philosophy

The project should be judged by its experimental rigor, not by claims of general intelligence.

Good evaluation questions include:

- Does training improve held-out eval performance?
- Does the system avoid simple echoing?
- Are confidence scores calibrated?
- Does memory consolidation improve or degrade recall?
- Can failure cases be inspected and explained?
- Are results reproducible with a fixed seed?

## Why this is relevant to AI systems roles

Nexus Cortex maps well to roles involving AI systems prototyping, research tooling, local model/runtime experimentation, evaluation harnesses, memory systems, applied research engineering, corpus processing, and instrumentation tools.

It is especially relevant as a demonstration of building AI infrastructure from first principles rather than only calling external model APIs.

## Current limitations

- Experimental architecture, not a production-grade LLM.
- Evaluation results need to be documented with reproducible benchmark outputs.
- Large corpora and generated model states are intentionally excluded from Git.
- More unit tests and CI should be added before public release.
- README screenshots and a short demo video should be added.

## Roadmap

- Add CI for `go test ./...` and `go vet ./...`.
- Add architecture diagram.
- Add reproducible benchmark report.
- Add small sample corpus and eval suite.
- Add dashboard screenshots.
- Add documentation for the core `cortex` package.
- Add `docs/LIMITATIONS.md` with clear experimental boundaries.

## Portfolio positioning

Use this project to demonstrate experimental AI systems thinking:

> I built a local bio-inspired AI research prototype in Go with corpus conversion, curriculum training, autonomous learning loops, evaluation suites, persistence, and a dashboard for introspection.

This is strongest when paired with `nexus-ai-clean`, which focuses more directly on local LLM runtime and model file tooling.
