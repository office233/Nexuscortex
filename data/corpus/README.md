# Pre-training Corpus Specifications

The pre-training corpus is used to build vocabulary, context sequences, and episodic knowledge prior to runtime evaluations.

## Data Formats

The corpus pipeline reads `.jsonl` files (JSON Lines). Each line represents a separate JSON object corresponding to one of two training modes:

### 1. Passive Text / Sequential Learning
Useful for loading large prose texts, articles, or books to learn language statistics, skip-grams, and general vocabulary.
```json
{
  "text": "câinele aleargă în parc."
}
```
Or:
```json
{
  "content": "pisica doarme pe canapea."
}
```

### 2. Associative Q&A Pairs
Used to explicitly encode deterministic facts into episodic memory (Hippocampus CA3 pattern completion) and reinforce Wernicke's sequence transitions.
```json
{
  "instruction": "ce face câinele?",
  "response": "câinele aleargă în parc."
}
```
Or:
```json
{
  "prompt": "where do neurons fire?",
  "completion": "neurons fire together, wire together."
}
```

## Curriculum Learning Pipeline

The training pipeline uses curriculum scheduling to sort inputs from simple (short token length) to complex (long paragraphs). This facilitates vocabulary stabilization and synaptic threshold growth before addressing longer contexts.
- **Dynamic Spaced Repetition:** Samples producing high surprise (prediction error) are re-queued automatically for further training cycles.
