package cortex

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"nexus-cortex/cortex/compute"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

// ─────────────────────────────────────────────────────────────────────
// organism.go — The Living Organism
// ─────────────────────────────────────────────────────────────────────
//
// The Organism is the central nervous system that wires every brain
// module into a single, coherent processing pipeline. It does NOT
// persist any direct answer maps — all behaviour emerges from the interplay
// of its subsystems:
//
//   Cerebellum  → fast-path cache (O(1) hash lookup)
//   Wernicke    → language understanding (text → SDR + keywords)
//   Hippocampus → episodic memory (SDR similarity recall)
//   Prefrontal  → deep thinking (spiking-network refinement)
//   Broca       → language generation (SDR → text via Brain chains)
//   Brain       → word-level associative memory (bigrams, skip-grams)
//
// Every call to Process() feeds the interaction back into all modules,
// so the organism grows smarter with every exchange.

// Configuration parameter defaults are defined in config.go.

// ─────────────────────────────────────────────────────────────────────
// Organism — The Living Brain
// ─────────────────────────────────────────────────────────────────────

// Organism is the central orchestrator that connects all brain modules
// into a single processing entity. It holds no pre-programmed knowledge;
// personality and capability emerge entirely from accumulated experience.
type Organism struct {
	Config Config // Central runtime configuration

	// Core modules — the foundation of all processing.
	Vocab   *Vocab
	Encoder *Encoder
	Decoder *Decoder
	Brain   *Brain

	// Brain regions — specialised subsystems.
	Wernicke       *Wernicke       // Language understanding
	Broca          *Broca          // Language generation
	Hippocampus    *Hippocampus    // Episodic memory
	SemanticMemory *SemanticMemory // Concept generalization
	Cerebellum     *Cerebellum     // Fast-path cache
	Prefrontal     *Prefrontal     // Deep thinking

	// Higher-order cognitive modules.
	Predictor         *Predictor          // Prediction & surprise
	Workspace         *Workspace          // Global workspace (conscious access)
	ThousandBrains    *ThousandBrains     // Multi-column voting
	Reward            *RewardSystem       // Reward / drive signal
	Emotion           *EmotionEngine      // Emotional state
	Self              *SelfModel          // Self-model / metacognition
	Curiosity         *CuriosityDrive     // Curiosity & exploration
	WorkingMem        *WorkingMemory      // Short-term working memory scratchpad
	Attention         *AttentionModule    // SDR self-attention (contextual bit weighting)
	ErrorLearner      *ErrorDrivenLearner // Error-driven synapse adjustment
	Analogy           *AnalogyEngine      // Analogical reasoning
	SleepConsolidator *SleepConsolidator  // Sleep-dependent memory replay
	Reasoning         *ReasoningEngine    // Deterministic symbolic reasoning (math, logic, sequences)

	// The massive, infinitely scaling parameter engine
	FractalCortex *FractalCortex // ALBERT-style shared stack (24× compression)

	// Radio frequency-based neural processor — neurons linked by resonance.
	// Replaces explicit synapses with 256-channel frequency bus.
	// Each neuron = 4 bytes RGBA32 (freq, phase, amplitude, routing).
	RadioCortex *RadioCortex
	SignalCodec *SignalCodec // Token ↔ frequency mapping

	// Broca 2.0 — transformer-based language model (optional).
	// When both are non-nil, Broca uses the transformer for generation.
	// When nil, falls back to Broca 1.0 (associative chain walker).
	Transformer *MiniTransformer // Autoregressive language model
	Tokenizer   *BPETokenizer   // BPE subword tokenizer

	// Body systems — sensory input, motor output, biological timing.
	Sensory *SensorySystem // Multi-channel sensory processing
	Motor   *MotorSystem   // Output filtering & queuing
	Rhythm  *RhythmEngine  // Biological clock & rhythms

	// Runtime state.
	InteractionCount uint64     // Atomically incremented on each Process() call
	Rng              *rand.Rand // Deterministic random source
}

// ─────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────

// NewOrganism creates a fully wired organism with the provided configuration.
// All modules start empty — the organism learns from experience.
func NewOrganism(cfg Config, rng *rand.Rand) *Organism {
	// Validate config before using it — fail fast on invalid values.
	if err := cfg.Validate(); err != nil {
		panic(fmt.Sprintf("invalid config: %v", err))
	}

	// Ensure the data directory exists.
	// Error is intentionally ignored: if the directory cannot be created,
	// subsequent file operations will fail with a clear error.
	_ = os.MkdirAll(cfg.DataDir, 0700)

	brainFile := filepath.Join(cfg.DataDir, "brain.nxbrain")
	vocabFile := filepath.Join(cfg.DataDir, "vocab.json")

	vocab := NewVocab()
	encoder := NewEncoder(vocab, cfg.SDRSize, cfg.ActiveCount, rng, cfg)
	decoder := NewDecoder(encoder, vocab)
	brain := NewBrain(brainFile, vocabFile, rng, cfg)
	// Share the same Vocab instance across Brain and the rest.
	brain.Vocab = vocab

	prefrontal := NewPrefrontal(cfg, rng)

	// Try hardware acceleration
	var engine interface{}
	gpu := compute.NewWebGPUEngine()
	gpu.Timeout = time.Duration(cfg.WebGPUTimeoutSecs) * time.Second
	if err := gpu.Init(); err == nil {
		fmt.Println("[GPU Compute] WebGPU Engine initialized successfully.")
		engine = gpu
	} else {
		fmt.Printf("[GPU Compute] Fallback to CPU Engine (WebGPU failed: %v)\n", err)
		engine = compute.NewCPUEngine()
	}

	return &Organism{
		Config:  cfg,
		Vocab:   vocab,
		Encoder: encoder,
		Decoder: decoder,
		Brain:   brain,

		Wernicke:       NewWernicke(vocab, encoder, cfg),
		Broca:          NewBroca(vocab, encoder, decoder, brain, cfg),
		Hippocampus:    NewHippocampus(cfg),
		SemanticMemory: NewSemanticMemory(cfg.SDRSize),
		Cerebellum:     NewCerebellum(cfg),
		Prefrontal:     prefrontal,

		Predictor:         NewPredictor(rng),
		Workspace:         NewWorkspace(cfg),
		ThousandBrains:    NewThousandBrains(cfg, rng),
		Reward:            NewRewardSystem(cfg),
		Emotion:           NewEmotionEngine(cfg),
		Self:              NewSelfModel(cfg),
		Curiosity:         NewCuriosityDrive(cfg),
		WorkingMem:        NewWorkingMemory(cfg),
		Attention:         NewAttentionModule(cfg),
		ErrorLearner:      NewErrorDrivenLearner(cfg),
		Analogy:           NewAnalogyEngine(cfg, brain, encoder),
		SleepConsolidator: NewSleepConsolidator(cfg),
		Reasoning:         NewReasoningEngine(),

		// FractalCortex: Infinite growing MoC ALBERT layers
		FractalCortex: NewFractalCortex(cfg, engine),

		// RadioCortex: frequency-based neural processor (configurable, default 1M neurons)
		RadioCortex: NewRadioCortex(cfg.RadioNeuronCount, rng),
		SignalCodec: NewSignalCodec(1000), // initial vocab, grows dynamically

		Sensory: NewSensorySystem(encoder, cfg),
		Motor:   NewMotorSystem(cfg),
		Rhythm:  NewRhythmEngine(cfg),

		Rng: rng,
	}
}

// ─────────────────────────────────────────────────────────────────────
// Process — THE MAIN PIPELINE
// ─────────────────────────────────────────────────────────────────────
//
// Process is the end-to-end cognitive loop. Nothing is hardcoded.
// The organism decides what to do based on its accumulated state:
//
//  1. CEREBELLUM CHECK (fast path)
//  2. UNDERSTAND       (Wernicke)
//  3. REMEMBER         (Hippocampus)
//  4. THINK            (Prefrontal)
//  5. SPEAK            (Broca)
//  6. LEARN            (feedback into all modules)
//
// Returns the generated response text, which may be empty if the
// organism has no knowledge to draw on.
func (o *Organism) Process(input string) string {
	atomic.AddUint64(&o.InteractionCount, 1)

	input = strings.TrimSpace(input)
	if input == "" {
		return ""
	}

	// ── RHYTHM: Advance the biological clock. ────────────────────
	o.Rhythm.Tick()

	// ── SENSORY: Process through sensory system. ────────────────
	sensoryOutput := o.Sensory.ProcessInput(input)

	// ── 2. UNDERSTAND (Wernicke) ─────────────────────────────────
	// Do this FIRST so we get a deterministic, noise-free SDR that
	// is used consistently by all modules (Cerebellum, Hippocampus,
	// Prefrontal).
	understanding := o.Wernicke.Understand(input)
	combinedSDR := understanding.Combined

	// ── REASONING: Deterministic symbolic check (arithmetic, logic, sequences). ──
	// This runs BEFORE the neural pipeline. If the input is a math problem,
	// sequence puzzle, syllogism, or sorting request, we return the exact answer
	// immediately instead of relying on SDR similarity matching.
	if o.Reasoning != nil {
		if reasonedAnswer, ok := o.Reasoning.TryReason(input); ok {
			// Deterministic answer — maximum confidence, skip neural pipeline.
			o.learnFromInteraction(input, reasonedAnswer, combinedSDR)
			o.WorkingMem.Store(combinedSDR, input, 255)
			o.WorkingMem.Tick()
			return reasonedAnswer
		}
	}

	// Union the sensory-derived SDR with Wernicke's combined SDR
	// to enrich the representation with multi-channel signals.
	combinedSDR = combinedSDR.Union(sensoryOutput.Combined)

	// ── ATTENTION: Sharpen the combined SDR via contextual relevance. ──
	// Observe the current combined SDR as context for future attention.
	o.Attention.ObserveContext(combinedSDR)
	// Compute per-bit relevance weights against recent context.
	attnWeights := o.Attention.ComputeWeights(combinedSDR, combinedSDR)
	attendedSDR := o.Attention.AttendSDR(combinedSDR, attnWeights)
	// Use the attended (sharpened) SDR if it retains enough active bits;
	// otherwise fall back to the unsharpened combinedSDR to avoid
	// degenerate empty representations.
	if attendedSDR.ActiveCount > 0 {
		combinedSDR = attendedSDR
	}

	// Learn n-gram context from this input immediately.
	o.Wernicke.LearnContext(understanding.Words)

	// ── PREDICT: What did we expect? Compute prediction error. ──
	predictionError := o.Predictor.Update(combinedSDR)
	// Make new prediction for next input.
	o.Predictor.Predict(combinedSDR)

	// ── ERROR-DRIVEN LEARNING: Adjust prefrontal synapses based on prediction error. ──
	if o.ErrorLearner != nil {
		o.ErrorLearner.AdjustSynapses(o.Prefrontal.Net, o.Predictor.LastPrediction, combinedSDR, predictionError)
	}

	// ── WORKSPACE: Submit to conscious awareness based on novelty. ──
	salience := predictionError // More surprising = more salient
	minAttention := uint8(o.Config.MinAttentionSalience)
	if salience < minAttention {
		salience = minAttention // Minimum attention
	}

	// Modulate salience by attention rhythm.
	attention := o.Rhythm.GetAttentionModulator()
	if attention < salience {
		salience = (salience + attention) / 2
	}

	o.Workspace.Submit(combinedSDR, salience)
	focusSDR := o.Workspace.Focus()

	// ── THOUSAND BRAINS: Process through voting columns. ──
	consensusSDR := o.ThousandBrains.Process(focusSDR)

	// ── REWARD: Convert prediction error to reward signal. ──
	reward := o.Reward.Surprise(predictionError)

	// ── EMOTION: Update emotional state. ──
	o.Emotion.Update(reward, predictionError, atomic.LoadUint64(&o.InteractionCount))

	// ── WORKING MEMORY: Recall recent context. ──
	// Check if working memory has relevant recent information.
	wmRecallText, wmSimilarity := o.WorkingMem.Recall(combinedSDR)
	_ = wmRecallText // Available for future context-aware generation

	// Blend working memory context into the input SDR.
	// This enriches the current input with traces of recent interactions,
	// enabling multi-turn reasoning.
	wmContext := o.WorkingMem.BlendContext(combinedSDR.Size)
	if wmContext.ActiveCount > 0 && wmSimilarity > 30 {
		// Partially merge WM context: keeps current input dominant.
		combinedSDR = combinedSDR.Union(wmContext)
	}

	// ── 1. CEREBELLUM CHECK ──────────────────────────────────────
	// Use the same deterministic combinedSDR as the cache key.
	if cached, ok := o.Cerebellum.Lookup(combinedSDR); ok {
		if cached.Confidence >= o.Config.CerebellumConfThreshold && cached.Text != "" {
			// Fast path hit — still learn from the interaction.
			o.learnFromInteraction(input, cached.Text, combinedSDR)
			o.WorkingMem.Store(combinedSDR, input, 180)
			o.WorkingMem.Tick()
			return cached.Text
		}
	}

	// ── 3. REMEMBER (Hippocampus) ────────────────────────────────
	var responseText string
	var responseSDR SDR
	memoryUsed := false
	var memorySimilarity uint8
	var memoryText string

	if mem, sim, ok := o.Hippocampus.RecallScored(combinedSDR, o.Config.HippocampusRecallThresh); ok {
		responseSDR = mem.Output
		memoryUsed = true
		memorySimilarity = sim
		memoryText = extractAnswerFromContext(mem.Context)
	}

	// Keyword-based fallback: if SDR recall failed or produced a
	// low-confidence match, try lexical keyword retrieval with Brain
	// association expansion. This compensates for the false-overlap
	// problem of union-encoded SDRs and enables matching rephrased
	// queries by expanding keywords through learned word associations.
	// Threshold=2 requires at least 2 stemmed keyword matches to
	// avoid false positives from single-stem coincidences.
	if !memoryUsed || memorySimilarity < o.Config.PrefrontalConfThreshold {
		inputTokens := Tokenize(input)
		// Calculate keyword threshold dynamically to prevent single-word false matches
		kwCount := 0
		for _, t := range inputTokens {
			t = strings.ToLower(t)
			if !hippoStopWords[t] && t != "" {
				kwCount++
			}
		}
		threshold := 1
		if kwCount >= 2 {
			threshold = 2
		}

		// Try expanded recall first (uses Brain associations for synonym expansion).
		if kwMem, kwScore, kwOK := o.Hippocampus.RecallByKeywordsExpanded(inputTokens, threshold, combinedSDR, o.Brain); kwOK && kwScore > memorySimilarity {
			responseSDR = kwMem.Output
			memoryUsed = true
			memorySimilarity = kwScore
			memoryText = extractAnswerFromContext(kwMem.Context)
		} else if kwMem, kwScore, kwOK := o.Hippocampus.RecallByKeywords(inputTokens, threshold, combinedSDR); kwOK && kwScore > memorySimilarity {
			// Fall back to non-expanded keyword recall.
			responseSDR = kwMem.Output
			memoryUsed = true
			memorySimilarity = kwScore
			memoryText = extractAnswerFromContext(kwMem.Context)
		}
	}

	var confidence uint8
	if memoryUsed {
		// Bypass Prefrontal refinement to prevent scrambling of deterministic facts retrieved from episodic memory.
		// Confidence must reflect match quality; a threshold hit is not proof of correctness.
		confidence = memorySimilarity
		o.Prefrontal.Confidence = memorySimilarity
	} else {
		// Enrich input with WM context for multi-turn reasoning.
		thinkInput := consensusSDR
		if wmContext.ActiveCount > 0 {
			thinkInput = thinkInput.Union(wmContext)
		}

		// Try Prefrontal first (fast spiking network).
		responseSDR = o.Prefrontal.ThinkDeep(thinkInput, o.Config.PrefrontalMaxHops)
		confidence = o.Prefrontal.GetConfidence()

		// ── RADIO CORTEX: Frequency-based processing via SignalCodec. ──
		// Convert input words to frequency chords and propagate through 100K neurons.
		if o.RadioCortex != nil && o.SignalCodec != nil {
			// Ensure codec covers current vocab
			if o.Vocab.Size() > o.SignalCodec.vocabSize {
				o.SignalCodec.GrowVocab(o.Vocab.Size())
			}

			// Encode input words as frequency chords
			var tokenIDs []int
			for _, w := range understanding.Words {
				if id := o.Vocab.Get(w); id > 0 {
					tokenIDs = append(tokenIDs, int(id))
				}
			}

			if len(tokenIDs) > 0 {
				// Clear and inject via codec (not raw SDR)
				o.RadioCortex.Bus.Clear()
				o.SignalCodec.EncodeTokens(&o.RadioCortex.Bus, tokenIDs, 200)

				// Activate input neurons
				for i := o.RadioCortex.InputStart; i < o.RadioCortex.InputEnd; i++ {
					n := &o.RadioCortex.Neurons[i]
					signal, busPhase := o.RadioCortex.Bus.Read(n.FreqListen())
					if signal > 0 {
						resonance := Resonance(n.Phase(), busPhase)
						if resonance > 20 {
							o.RadioCortex.Fired[i] = true
						}
					}
				}

				// 20 ticks — enough for input → hidden → hidden → output
				for tick := 0; tick < 20; tick++ {
					o.RadioCortex.Step()
				}

				// After training period, blend Radio output
				if o.RadioCortex.TickCount > 1000 && confidence < o.Config.PrefrontalConfThreshold {
					radioSDR := o.RadioCortex.ReadOutputSDR(combinedSDR.Size)
					if radioSDR.ActiveCount > 5 {
						filtered := responseSDR.Intersect(radioSDR)
						if filtered.ActiveCount > 0 {
							responseSDR = filtered
						}
					}
				}
			}
		}

		// If Prefrontal is not confident, escalate to FractalCortex (infinite dynamic scale).
		// The FractalCortex routes through dynamically growing ALBERT-shared layers of ternary weights
		// for deeper reasoning than the spiking network can provide.
		if confidence < o.Config.PrefrontalConfThreshold && o.FractalCortex != nil {
			deepSDR := o.FractalCortex.ProcessToken(thinkInput)

			if deepSDR.ActiveCount > 0 {
				responseSDR = deepSDR
				// Recompute confidence from FractalCortex output quality
				overlap := sdrAnd(thinkInput, deepSDR)

				errorMagnitude := 1.0
				if thinkInput.ActiveCount > 0 {
					errorMagnitude = 1.0 - float64(overlap.ActiveCount)/float64(thinkInput.ActiveCount)
				}

				confidence = uint8((1.0 - errorMagnitude) * 255.0)

				// Dynamic growth trigger — spawn new cortex block on high error
				// If error is high, this concept is alien. Spawn new physical parameters!
				o.FractalCortex.CheckPredictionError(errorMagnitude)
			}
		}
	}

	// ── 5. SPEAK (Broca) ─────────────────────────────────────────
	// Priority 0: Broca 2.0 — Transformer-based autoregressive generation
	// Uses BPE tokenizer + trained MiniTransformer for fluent text.
	if o.Transformer != nil && o.Tokenizer != nil && len(understanding.Words) > 0 {
		mem := ""
		if memoryUsed && memoryText != "" {
			mem = memoryText
		}
		responseText = o.Broca.GenerateWithTransformer(
			o.Transformer, o.Tokenizer,
			understanding.Words, mem,
			confidence, o.Config.MaxGenWords,
		)
	}

	// ── PRIORITY 0: Hippocampus Direct Recall ──────────────────────
	// If we have a stored memory matching this query, USE IT directly.
	// Memory recall > generation. A human remembers facts before "generating" them.
	if responseText == "" && memoryUsed && memoryText != "" && !sameText(memoryText, input) {
		// Extract the answer part from "question | answer" format
		answer := extractAnswerFromContext(memoryText)
		if answer != "" && !sameText(answer, input) {
			responseText = answer
		}
	}

	// Priority 1: Broca 1.0 — FractalCortex autoregressive (SDR-based)
	// Generate a response autoregressively through FractalCortex ternary layers.
	// If Hippocampus retrieved a memory, inject it as context (RAG-style).
	if responseText == "" && o.FractalCortex != nil && len(understanding.Words) > 0 {
		contextWords := make([]string, 0, len(understanding.Words)+10)
		if memoryUsed && memoryText != "" {
			// RAG context injection
			contextWords = append(contextWords, Tokenize(memoryText)...)
			contextWords = append(contextWords, "|")
		}
		contextWords = append(contextWords, understanding.Words...)
		
		responseText = o.Broca.GenerateAutoregressive(o.FractalCortex, contextWords, o.Config.MaxGenWords)
	}

	// Anti-repetition filter: if any generator produces "word word word word...",
	// that's a degenerate loop, not a real response. Clear it and try the next.
	if isRepetitive(responseText) {
		responseText = ""
	}

	// Priority 2: RadioCortex autoregressive generation (frequency-based)
	// Uses SignalCodec to decode bus spectrum into tokens.
	if responseText == "" && o.RadioCortex != nil && o.SignalCodec != nil && o.RadioCortex.TickCount > 2000 {
		var tokenIDs []int
		for _, w := range understanding.Words {
			if id := o.Vocab.Get(w); id > 0 {
				tokenIDs = append(tokenIDs, int(id))
			}
		}
		if len(tokenIDs) > 0 {
			radioText := o.RadioCortex.RadioGenerate(o.SignalCodec, o.Vocab, tokenIDs, o.Config.MaxGenWords, 20)
			if radioText != "" && !isRepetitive(radioText) {
				responseText = radioText
			}
		}
	}

	// Fallback to associative / SDR generation if autoregression didn't work or wasn't used
	if responseText == "" {
		candidate := o.Broca.Generate(responseSDR, o.Config.MaxGenWords)
		if !isRepetitive(candidate) {
			responseText = candidate
		}
	}

	if responseText == "" && len(understanding.KeyWords) > 0 {
		candidate := o.Broca.GenerateFromContext(understanding.KeyWords, o.Config.MaxGenWords)
		if !isRepetitive(candidate) {
			responseText = candidate
		}
	}
	if responseText == "" && len(understanding.Words) > 0 {
		candidate := o.Broca.GenerateFromContext(understanding.Words, o.Config.MaxGenWords)
		if !isRepetitive(candidate) {
			responseText = candidate
		}
	}

	// Don't echo the input back as a response.
	if sameText(responseText, input) {
		responseText = ""
	}
	if responseText != "" && confidence < o.Config.PrefrontalConfThreshold {
		responseText = ""
	}

	// ── 6. LEARN ─────────────────────────────────────────────────
	// Store the Prefrontal-refined SDR in Hippocampus (not re-encoded) if we have a valid response.
	// This preserves the neural computation's output.
	if responseText != "" && responseText != "(no confident response)" && combinedSDR.ActiveCount > 0 && responseSDR.ActiveCount > 0 {
		o.Hippocampus.Store(combinedSDR, responseSDR, input)
	}

	// Cache in Cerebellum if confidence is high enough.
	if confidence >= o.Config.PrefrontalConfThreshold && responseText != "" {
		o.Cerebellum.Learn(combinedSDR, responseSDR, responseText, confidence)
	}

	// Feed the interaction into Brain's word-level associations.
	o.Brain.Learn(input)
	if responseText != "" {
		o.Brain.Learn(responseText)
	}

	// Feed response words into Wernicke's n-gram tracker.
	if responseText != "" {
		responseTokens := Tokenize(responseText)
		o.Wernicke.LearnContext(responseTokens)
	}

	// ── RADIO HEBBIAN: Reinforce/weaken neurons based on response quality. ──
	if o.RadioCortex != nil {
		if responseText != "" && responseText != "(no confident response)" && confidence >= o.Config.PrefrontalConfThreshold {
			o.RadioCortex.Confirm() // Good response → strengthen fired neurons
		} else {
			o.RadioCortex.Contradict() // Bad/no response → weaken and re-tune
		}
	}

	// ── SELF: Track competence. ──────────────────────────────────
	topic := ""
	if len(understanding.KeyWords) > 0 {
		topic = understanding.KeyWords[0]
	}
	if responseText != "" && responseText != "(no confident response)" && confidence >= o.Config.PrefrontalConfThreshold {
		o.Self.RecordSuccess(topic, confidence)
	} else {
		o.Self.RecordFailure(topic)
	}

	// ── CURIOSITY: Observe error for this topic. ─────────────────
	o.Curiosity.ObserveError(predictionError, topic)

	// ── MOTOR: Route output through motor system. ───────────────
	if responseText != "" {
		motorConf := confidence
		o.Motor.Enqueue(MotorCommand{
			Text:       responseText,
			Confidence: motorConf,
			Urgency:    o.Rhythm.GetSleepPressure(), // More tired = less urgent
			Source:     "process",
		})
		if cmd, ok := o.Motor.Execute(); ok {
			responseText = cmd.Text // Motor may filter/reorder
		}
	}

	if responseText == "" {
		responseText = "(no confident response)"
	}

	// ── WORKING MEMORY: Store this interaction for next-turn context. ──
	// Store the input SDR with moderate relevance.
	o.WorkingMem.Store(combinedSDR, input, 150+confidence/4)
	// Store the response SDR with confidence-proportional relevance.
	if responseText != "(no confident response)" && responseSDR.ActiveCount > 0 {
		o.WorkingMem.Store(responseSDR, responseText, confidence)
	}
	// Age all WM slots — irrelevant items will fade automatically.
	o.WorkingMem.Tick()

	return responseText
}

// learnFromInteraction is a lightweight learning step used when the
// cerebellum fast-path fires. It still updates frequency counters
// without the full pipeline cost.
func (o *Organism) learnFromInteraction(input, response string, inputSDR SDR) {
	o.Brain.Learn(input)
	if response != "" {
		o.Brain.Learn(response)
	}
	tokens := Tokenize(input)
	o.Wernicke.LearnContext(tokens)
	if response != "" {
		o.Wernicke.LearnContext(Tokenize(response))
	}
}

// ─────────────────────────────────────────────────────────────────────
// Learn — Passive knowledge absorption
// ─────────────────────────────────────────────────────────────────────

// Learn feeds text into the organism without expecting a response.
// Use this to pre-train the organism on a corpus of text.
//
// The text is absorbed by:
//   - Brain:       word-level bigram and skip-gram associations
//   - Wernicke:    n-gram frequency tracking
//   - Hippocampus: stored as an episodic memory (input SDR → input SDR)
func (o *Organism) Learn(text string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}

	// Brain: word associations.
	o.Brain.Learn(text)

	// Wernicke: n-gram context.
	tokens := Tokenize(text)
	o.Wernicke.LearnContext(tokens)

	// Hippocampus: encode and store as a self-referential memory.
	// The input and output SDRs are the same — this memory records
	// the experience of encountering this text.
	sdr := o.Encoder.EncodeSentence(text)
	o.Hippocampus.Store(sdr, sdr, text)
}

// LearnQA absorbs supervised question-answer examples into neural,
// associative, and episodic memory. It deliberately does not persist a
// direct question -> answer lookup table; recall is cued by SDR
// similarity through the hippocampus.
//
// The hippocampus context stores "question | answer" so the keyword
// index covers vocabulary from both sides, enabling generalization
// queries that use question-adjacent words to locate the memory.
func (o *Organism) LearnQA(question, answer string) {
	question = strings.TrimSpace(question)
	answer = strings.TrimSpace(answer)
	if question == "" || answer == "" {
		return
	}

	joined := strings.TrimSpace(question + " " + answer)
	o.Brain.Learn(joined)
	o.Brain.Learn(question)
	o.Brain.Learn(answer)

	o.Wernicke.LearnContext(Tokenize(question))
	o.Wernicke.LearnContext(Tokenize(answer))
	o.Wernicke.LearnContext(Tokenize(joined))

	questionSDR := o.Encoder.EncodeSentence(question)
	answerSDR := o.Encoder.EncodeSentence(answer)
	// Store "Q | A" as context so the keyword index covers both.
	context := question + " | " + answer
	o.Hippocampus.Store(questionSDR, answerSDR, context)

	// FractalCortex Autoregressive Training (STDP)
	if o.FractalCortex != nil {
		// Encode the full sequence (Q | A) token by token to create a sequence array
		// We add "|" as a separator token so the cortex learns to transition from Q to A
		seqTokens := append(Tokenize(question), "|")
		seqTokens = append(seqTokens, Tokenize(answer)...)
		
		seqSDRs := make([]SDR, 0, len(seqTokens))
		for _, tok := range seqTokens {
			seqSDRs = append(seqSDRs, o.Encoder.EncodeWord(tok))
		}
		
		// Train the cortex using Probabilistic STDP with a moderate learning rate (e.g., 20/255 = ~8% probability)
		o.FractalCortex.TrainSequence(seqSDRs, 20)
	}

	// RadioCortex: Train frequency associations (Q → A)
	// Each answer token becomes a training target: input=question, target=answer_token
	if o.RadioCortex != nil && o.SignalCodec != nil {
		// Ensure codec covers current vocab
		if o.Vocab.Size() > o.SignalCodec.vocabSize {
			o.SignalCodec.GrowVocab(o.Vocab.Size())
		}

		// Convert question words to token IDs
		qTokens := Tokenize(question)
		var qIDs []int
		for _, w := range qTokens {
			if id := o.Vocab.Get(w); id > 0 {
				qIDs = append(qIDs, int(id))
			}
		}

		// Train on each answer token as target
		aTokens := Tokenize(answer)
		for _, w := range aTokens {
			if id := o.Vocab.Get(w); id > 0 {
				// Train: given question frequencies → produce this answer token's frequencies
				o.RadioCortex.RadioTrainStep(o.SignalCodec, qIDs, int(id), 20)
			}
		}
	}
}

func sameText(a, b string) bool {
	return strings.EqualFold(strings.TrimSpace(a), strings.TrimSpace(b))
}

// isRepetitive detects degenerate output like "brain brain brain brain...".
// Returns true if >50% of the words are the same word repeated.
func isRepetitive(text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	words := strings.Fields(text)
	if len(words) <= 2 {
		return false // too short to judge
	}

	// Count word frequencies
	counts := make(map[string]int, len(words))
	for _, w := range words {
		counts[strings.ToLower(w)]++
	}

	// Find the most frequent word
	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}

	// If one word is >50% of all words, it's repetitive
	return float64(maxCount)/float64(len(words)) > 0.5
}

// extractAnswerFromContext extracts the answer part from a hippocampus
// context string that was stored in "question | answer" format. If the
// separator is not found, the whole string is returned (backward
// compatible with older memories that stored only the answer).
func extractAnswerFromContext(context string) string {
	if idx := strings.Index(context, " | "); idx >= 0 {
		return strings.TrimSpace(context[idx+3:])
	}
	return strings.TrimSpace(context)
}

// ─────────────────────────────────────────────────────────────────────
// Sleep — Offline maintenance and consolidation
// ─────────────────────────────────────────────────────────────────────

// Sleep performs offline maintenance, analogous to sleep-dependent
// memory consolidation in biological brains:
//
//  1. Hippocampus.Consolidate():  strengthen strong memories, prune weak
//  2. Brain.Prune():              forget old, unused word associations
//  3. Cerebellum.Prune():         evict unused cache entries
//  4. Prefrontal.Net.PruneWeak(): remove weak synapses from the network
func (o *Organism) Sleep() []string {
	// 0. Semantic memory generalization (neocortical abstraction from episodic memories)
	if o.SemanticMemory != nil {
		o.SemanticMemory.Generalize(o.Hippocampus)
	}

	// 0.25. Sleep consolidation: replay hippocampal memories through prefrontal.
	var consolidationLogs []string
	if o.SleepConsolidator != nil {
		report := o.SleepConsolidator.Consolidate(
			o.Hippocampus, o.Prefrontal, o.Brain, o.Broca, o.Self, o.Config)
		consolidationLogs = report.Logs
	}

	// 0.5. Active prefrontal self-reflection training
	_, _, logs := o.SelfTrain()

	// Merge consolidation logs before self-train logs.
	logs = append(consolidationLogs, logs...)

	// 1.5. Broca 2.0 self-evolution: replay memories through Transformer.
	// This is the key differentiator — the organism learns from its own
	// experience during sleep, getting better at language over time.
	if o.Transformer != nil && o.Tokenizer != nil {
		memTrained, corpusTrained, avgLoss := o.SelfEvolve()
		if memTrained+corpusTrained > 0 {
			logs = append(logs, fmt.Sprintf(
				"[SelfEvolve] Trained Broca 2.0: %d memories + %d corpus lines (avg loss: %.4f)",
				memTrained, corpusTrained, avgLoss))
		}
	}

	// 2. Hippocampal consolidation (LTP stabilisation + decay).
	o.Hippocampus.Consolidate()

	// 2. Brain pruning (forget old associations).
	o.Brain.Prune(o.Config.BrainPruneMaxAge)

	// 3. Cerebellum pruning (evict rarely-used cache entries).
	o.Cerebellum.Prune(o.Config.CerebellumMinUseCount)

	// 4. Prefrontal network pruning (remove weak reservoir synapses).
	o.Prefrontal.Net.PruneWeak(o.Config.PrefrontalPruneThresh)

	// 5. Emotion decay (mood normalisation).
	o.Emotion.Decay()

	// 6. Curiosity decay (interest fading).
	o.Curiosity.Decay()

	// 7. Rhythm reset (sleep resets the biological clock).
	o.Rhythm.OnSleep()

	// 8. FractalCortex: reset growth lock to re-enable neurogenesis.
	if o.FractalCortex != nil {
		o.FractalCortex.ResetGrowthLock()

		// 8.5. Merge pending PlasticityJournal entries into ternary weights.
		if o.FractalCortex.Journal != nil && o.FractalCortex.Journal.Size() > 0 {
			applied := o.FractalCortex.MergeJournal()
			if applied > 0 {
				logs = append(logs, fmt.Sprintf("[QuantumPlasticity] Merged %d journal entries into ternary weights.", applied))
			}
		}
	}

	// 9. RadioCortex: neurogenesis — replace dead neurons with fresh ones.
	if o.RadioCortex != nil {
		replaced := o.RadioCortex.Neurogenesis()
		if replaced > 0 {
			logs = append(logs, fmt.Sprintf("[RadioCortex] Neurogenesis: replaced %d dead neurons.", replaced))
		}
	}

	// 10. Meta-learning: queue weak topics for auto-exploration.
	autoLearnTopics := o.AutoLearn()
	for _, topic := range autoLearnTopics {
		logs = append(logs, fmt.Sprintf("AutoLearn: queued weak topic '%s' for curiosity-driven exploration.", topic))
	}

	return logs
}

// ─────────────────────────────────────────────────────────────────────
// AutoLearn — Meta-Learning (C.8)
// ─────────────────────────────────────────────────────────────────────
//
// AutoLearn connects the SelfModel's awareness of weak topics to the
// CuriosityDrive, creating an automatic meta-learning loop. For each
// topic where the organism knows it is weak, if the CuriosityDrive
// says exploration is warranted, the topic is boosted with a high
// error signal to focus future curiosity on that gap.
//
// Returns the list of topics queued for exploration.
func (o *Organism) AutoLearn() []string {
	weakTopics := o.Self.WeakTopics()
	if len(weakTopics) == 0 {
		return nil
	}

	var queued []string
	for _, topic := range weakTopics {
		if topic == "" {
			continue
		}
		// Only queue topics for exploration if the curiosity drive says explore.
		if !o.Curiosity.ShouldExplore() {
			continue
		}
		// Boost interest in this weak topic by injecting a high prediction
		// error signal (200), pushing it into the "learnable zone" of curiosity.
		o.Curiosity.ObserveError(200, topic)
		queued = append(queued, topic)
	}
	return queued
}

// ─────────────────────────────────────────────────────────────────────
// Stats — Comprehensive organism introspection
// ─────────────────────────────────────────────────────────────────────

// OrganismStats holds comprehensive statistics from all brain modules.
type OrganismStats struct {
	InteractionCount uint64 `json:"interaction_count"`

	// Vocabulary.
	VocabSize int `json:"vocab_size"`

	// Brain (word-level associations).
	BrainSynapses       int    `json:"brain_synapses"`
	BrainActiveSynapses int    `json:"brain_active_synapses"`
	BrainAvgWeight      uint32 `json:"brain_avg_weight"` // ×256

	// Hippocampus (episodic memory).
	HippocampusMemories    int `json:"hippocampus_memories"`
	HippocampusMaxMemories int `json:"hippocampus_max_memories"`

	// Cerebellum (cache).
	CerebellumHits      uint64 `json:"cerebellum_hits"`
	CerebellumMisses    uint64 `json:"cerebellum_misses"`
	CerebellumCacheSize int    `json:"cerebellum_cache_size"`
	CerebellumCached    int    `json:"cerebellum_cached"` // alias for CacheSize

	// Prefrontal (reasoning network).
	PrefrontalNeurons    int   `json:"prefrontal_neurons"`
	PrefrontalSynapses   int   `json:"prefrontal_synapses"`
	PrefrontalConfidence uint8 `json:"prefrontal_confidence"`
	PrefrontalGoals      int   `json:"prefrontal_goals"` // = active think cycles

	// Wernicke (language understanding).
	WernickeNGrams int `json:"wernicke_ngrams"`
	WernickeRules  int `json:"wernicke_rules"` // alias for NGrams

	// Broca (language generation).
	BrocaPatterns int `json:"broca_patterns"` // = Brain active synapses

	// Higher-order cognitive stats.
	PredictionError            uint8  `json:"prediction_error"`
	SurpriseLevel              uint8  `json:"surprise_level"`
	EmotionalMood              string `json:"emotional_mood"`
	Valence                    int8   `json:"valence"`
	Arousal                    uint8  `json:"arousal"`
	CuriosityLevel             uint8  `json:"curiosity_level"`
	ExplorationRate            uint8  `json:"exploration_rate"`
	SelfAccuracy               uint8  `json:"self_accuracy"`
	ThousandBrainsDisagreement uint8  `json:"thousand_brains_disagreement"`
	RewardDrive                int8   `json:"reward_drive"`
	IsInFlow                   bool   `json:"is_in_flow"`

	// Body systems.
	SleepPressure uint8  `json:"sleep_pressure"`
	Alertness     uint8  `json:"alertness"`
	NeedsSleep    bool   `json:"needs_sleep"`
	SensoryInputs uint64 `json:"sensory_inputs"`
	MotorOutputs  uint64 `json:"motor_outputs"`
	RhythmTick    uint64 `json:"rhythm_tick"`

	// RadioCortex (frequency-based neural processor).
	RadioNeurons   int    `json:"radio_neurons"`
	RadioAlive     int    `json:"radio_alive"`
	RadioAvgAmp    int    `json:"radio_avg_amplitude"`
	RadioTickCount uint64 `json:"radio_tick_count"`

	// Aggregate.
	TotalSynapticWeight int64 `json:"total_synaptic_weight"`
}

// Stats returns a snapshot of every module's current state.
func (o *Organism) Stats() OrganismStats {
	brainStats := o.Brain.GetStats()
	cbHits, cbMisses, cbSize := o.Cerebellum.Stats()
	netStats := o.Prefrontal.Net.Stats()

	return OrganismStats{
		InteractionCount: atomic.LoadUint64(&o.InteractionCount),

		VocabSize: o.Vocab.Size(),

		BrainSynapses:       brainStats.TotalSynapses,
		BrainActiveSynapses: brainStats.ActiveSynapses,
		BrainAvgWeight:      brainStats.AvgWeight,

		HippocampusMemories:    o.Hippocampus.Size(),
		HippocampusMaxMemories: o.Hippocampus.MaxMemories,

		CerebellumHits:      cbHits,
		CerebellumMisses:    cbMisses,
		CerebellumCacheSize: cbSize,
		CerebellumCached:    cbSize,

		PrefrontalNeurons:    netStats.TotalNeurons,
		PrefrontalSynapses:   netStats.TotalSynapses,
		PrefrontalConfidence: o.Prefrontal.GetConfidence(),
		PrefrontalGoals:      o.Prefrontal.ThinkCycles,

		WernickeNGrams: len(o.Wernicke.NGrams),
		WernickeRules:  len(o.Wernicke.NGrams),

		BrocaPatterns: brainStats.ActiveSynapses,

		PredictionError: o.Predictor.GetSurprise(),
		SurpriseLevel:   o.Predictor.GetSurprise(),
		EmotionalMood:   o.Emotion.GetMood(),
		Valence:         o.Emotion.GetState().Valence,
		Arousal:         o.Emotion.GetState().Arousal,
		CuriosityLevel:  func() uint8 { _, c := o.Curiosity.MostInteresting(); return c }(),
		ExplorationRate: func() uint8 {
			if o.Curiosity.ShouldExplore() {
				return 255
			}
			return 0
		}(),
		SelfAccuracy:               o.Self.GetAccuracy(),
		ThousandBrainsDisagreement: o.ThousandBrains.Disagree(),
		RewardDrive:                o.Reward.GetDrive(),
		IsInFlow:                   o.Reward.IsInFlow(),

		SleepPressure: o.Rhythm.GetSleepPressure(),
		Alertness:     o.Rhythm.GetAlertness(),
		NeedsSleep:    o.Rhythm.NeedsSleep(),
		SensoryInputs: o.Sensory.Stats().TotalInputs,
		MotorOutputs:  o.Motor.Stats().TotalOutputs,
		RhythmTick:    o.Rhythm.Stats().GlobalTick,

		RadioNeurons:   func() int { if o.RadioCortex != nil { return o.RadioCortex.Stats().TotalNeurons }; return 0 }(),
		RadioAlive:     func() int { if o.RadioCortex != nil { return o.RadioCortex.Stats().AliveNeurons }; return 0 }(),
		RadioAvgAmp:    func() int { if o.RadioCortex != nil { return o.RadioCortex.Stats().AvgAmplitude }; return 0 }(),
		RadioTickCount: func() uint64 { if o.RadioCortex != nil { return o.RadioCortex.Stats().TickCount }; return 0 }(),

		TotalSynapticWeight: int64(brainStats.AvgWeight) * int64(brainStats.ActiveSynapses) / 256,
	}
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save / Load
// ─────────────────────────────────────────────────────────────────────
//
// The organism is serialized as a directory of files:
//
//   <dataDir>/
//     brain.nxbrain      — word-level synapses (binary)
//     vocab.json         — word↔ID mappings
//     encoder.json       — SDR mappings
//     hippocampus.nxhip  — episodic memories (binary)
//     network.nxnet      — prefrontal spiking network (binary)
//     organism.json      — runtime metadata
//
// The Cerebellum cache is intentionally NOT persisted. It is a
// volatile fast-path that is rebuilt naturally as the organism
// encounters repeated inputs.

// organismMeta is the JSON-serializable metadata envelope.
type organismMeta struct {
	InteractionCount uint64 `json:"interaction_count"`
}

// Save persists all organism state to the given directory.
// The directory is created if it does not exist.
func (o *Organism) Save(dataDir string) error {
	if err := os.MkdirAll(dataDir, 0700); err != nil {
		return fmt.Errorf("organism save mkdir: %w", err)
	}

	// 1. Vocab.
	vocabPath := filepath.Join(dataDir, "vocab.json")
	if err := o.Vocab.Save(vocabPath); err != nil {
		return fmt.Errorf("organism save vocab: %w", err)
	}

	// 2. Encoder.
	encoderPath := filepath.Join(dataDir, "encoder.json")
	if err := o.Encoder.Save(encoderPath); err != nil {
		return fmt.Errorf("organism save encoder: %w", err)
	}

	// 3. Brain (updates its own paths to match dataDir).
	o.Brain.FilePath = filepath.Join(dataDir, "brain.nxbrain")
	o.Brain.VocabPath = vocabPath
	if err := o.Brain.Save(); err != nil {
		return fmt.Errorf("organism save brain: %w", err)
	}

	// 4. Hippocampus.
	hipPath := filepath.Join(dataDir, "hippocampus.nxhip")
	if err := o.Hippocampus.Save(hipPath); err != nil {
		return fmt.Errorf("organism save hippocampus: %w", err)
	}

	// 4.5. Semantic memory generalization state.
	semPath := filepath.Join(dataDir, "semantic.json")
	if err := o.SemanticMemory.Save(semPath); err != nil {
		return fmt.Errorf("organism save semantic memory: %w", err)
	}

	// 5. Prefrontal network.
	netPath := filepath.Join(dataDir, "network.nxnet")
	if err := o.Prefrontal.Net.Save(netPath); err != nil {
		return fmt.Errorf("organism save prefrontal: %w", err)
	}

	// 6. Organism metadata.
	meta := organismMeta{
		InteractionCount: atomic.LoadUint64(&o.InteractionCount),
	}
	metaJSON, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("organism save meta marshal: %w", err)
	}
	metaPath := filepath.Join(dataDir, "organism.json")
	if err := os.WriteFile(metaPath, metaJSON, 0600); err != nil {
		return fmt.Errorf("organism save meta write: %w", err)
	}

	// 7. Wernicke (n-gram context).
	wernickePath := filepath.Join(dataDir, "wernicke.json")
	if err := o.Wernicke.Save(wernickePath); err != nil {
		return fmt.Errorf("organism save wernicke: %w", err)
	}

	// 8. Emotion (mood/valence/arousal).
	emotionPath := filepath.Join(dataDir, "emotion.json")
	if err := o.Emotion.Save(emotionPath); err != nil {
		return fmt.Errorf("organism save emotion: %w", err)
	}

	// 9. Self (competence tracking).
	selfPath := filepath.Join(dataDir, "self.json")
	if err := o.Self.Save(selfPath); err != nil {
		return fmt.Errorf("organism save self: %w", err)
	}

	// 10. Curiosity (exploration state).
	curiosityPath := filepath.Join(dataDir, "curiosity.json")
	if err := o.Curiosity.Save(curiosityPath); err != nil {
		return fmt.Errorf("organism save curiosity: %w", err)
	}

	// 11. Reward (drive state).
	rewardPath := filepath.Join(dataDir, "reward.json")
	if err := o.Reward.Save(rewardPath); err != nil {
		return fmt.Errorf("organism save reward: %w", err)
	}

	// 12. FractalCortex weights.
	if o.FractalCortex != nil {
		if err := o.FractalCortex.Save(dataDir); err != nil {
			return fmt.Errorf("organism save fractal cortex: %w", err)
		}
	}

	// 13. RadioCortex neurons + SignalCodec vocab size.
	if o.RadioCortex != nil {
		if err := SaveRadioCortexWithCodec(o.RadioCortex, o.SignalCodec, filepath.Join(dataDir, "radio_cortex.nxrc")); err != nil {
			return fmt.Errorf("organism save radio cortex: %w", err)
		}
	}

	return nil
}

// LoadOrganism restores a fully wired organism from a previously
// saved data directory. All module interconnections are re-established
// so the organism can resume processing immediately.
func LoadOrganism(cfg Config, rng *rand.Rand) (*Organism, error) {
	// 1. Vocab.
	vocabPath := filepath.Join(cfg.DataDir, "vocab.json")
	vocab, err := LoadVocab(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("organism load vocab: %w", err)
	}

	// 2. Encoder (reconnected to the loaded vocab).
	encoderPath := filepath.Join(cfg.DataDir, "encoder.json")
	encoder, err := LoadEncoder(encoderPath, vocab, rng)
	if err != nil {
		return nil, fmt.Errorf("organism load encoder: %w", err)
	}

	// 3. Decoder (reconnected to encoder + vocab).
	decoder := NewDecoder(encoder, vocab)

	// 4. Brain.
	brainPath := filepath.Join(cfg.DataDir, "brain.nxbrain")
	brain, err := LoadBrain(brainPath, vocabPath, rng, cfg)
	if err != nil {
		return nil, fmt.Errorf("organism load brain: %w", err)
	}
	// Ensure the Brain uses the shared Vocab instance.
	brain.Vocab = vocab

	// 5. Hippocampus.
	hipPath := filepath.Join(cfg.DataDir, "hippocampus.nxhip")
	hippocampus, err := LoadHippocampus(hipPath)
	if err != nil {
		return nil, fmt.Errorf("organism load hippocampus: %w", err)
	}
	hippocampus.MaxMemories = cfg.MaxMemories
	hippocampus.ReconsolidationThresh = cfg.HippoReconsolidationThresh
	hippocampus.InitialStrength = cfg.HippoInitialStrength
	hippocampus.LtpThreshold = cfg.HippoLtpThreshold

	// 5.5. Semantic memory generalization state — non-fatal fallback if missing.
	semPath := filepath.Join(cfg.DataDir, "semantic.json")
	semanticMemory, err := LoadSemanticMemory(semPath, cfg.SDRSize)
	if err != nil {
		semanticMemory = NewSemanticMemory(cfg.SDRSize)
	}

	// 6. Prefrontal network.
	netPath := filepath.Join(cfg.DataDir, "network.nxnet")
	net, err := LoadNetwork(netPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("organism load prefrontal: %w", err)
	}
	prefrontal := &Prefrontal{
		Net:          net,
		ThinkCycles:  cfg.ThinkCycles,
		Confidence:   0,
		InputCurrent: cfg.PrefrontalInputCurrent,
		Cfg:          cfg,
	}

	// 7. Organism metadata.
	var interactionCount uint64
	metaPath := filepath.Join(cfg.DataDir, "organism.json")
	if metaRaw, err := os.ReadFile(metaPath); err == nil {
		var meta organismMeta
		if err := json.Unmarshal(metaRaw, &meta); err == nil {
			interactionCount = meta.InteractionCount
		}
	}
	// Metadata file missing is non-fatal — the organism just loses
	// its interaction counter.

	// 8. Wernicke (n-gram context) — fall back to fresh if missing.
	wernickePath := filepath.Join(cfg.DataDir, "wernicke.json")
	wernicke, err := LoadWernicke(wernickePath, vocab, encoder, cfg)
	if err != nil {
		return nil, fmt.Errorf("organism load wernicke: %w", err)
	}
	if wernicke == nil {
		wernicke = NewWernicke(vocab, encoder, cfg)
	}

	// 9. Emotion (mood/valence/arousal) — fall back to fresh if missing.
	emotionPath := filepath.Join(cfg.DataDir, "emotion.json")
	emotion, err := LoadEmotionEngine(emotionPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("organism load emotion: %w", err)
	}
	if emotion == nil {
		emotion = NewEmotionEngine(cfg)
	}

	// 10. Self (competence tracking) — fall back to fresh if missing.
	selfPath := filepath.Join(cfg.DataDir, "self.json")
	self, err := LoadSelfModel(selfPath)
	if err != nil {
		return nil, fmt.Errorf("organism load self: %w", err)
	}
	if self == nil {
		self = NewSelfModel(cfg)
	}
	self.Cfg = cfg

	// 11. Curiosity (exploration state) — fall back to fresh if missing.
	curiosityPath := filepath.Join(cfg.DataDir, "curiosity.json")
	curiosity, err := LoadCuriosityDrive(curiosityPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("organism load curiosity: %w", err)
	}
	if curiosity == nil {
		curiosity = NewCuriosityDrive(cfg)
	}

	// 12. Reward (drive state) — fall back to fresh if missing.
	rewardPath := filepath.Join(cfg.DataDir, "reward.json")
	reward, err := LoadRewardSystem(rewardPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("organism load reward: %w", err)
	}
	if reward == nil {
		reward = NewRewardSystem(cfg)
	}

	// Try hardware acceleration
	var engine interface{}
	gpu := compute.NewWebGPUEngine()
	gpu.Timeout = time.Duration(cfg.WebGPUTimeoutSecs) * time.Second
	if err := gpu.Init(); err == nil {
		fmt.Println("[GPU Compute] WebGPU Engine initialized successfully.")
		engine = gpu
	} else {
		fmt.Printf("[GPU Compute] Fallback to CPU Engine (WebGPU failed: %v)\n", err)
		engine = compute.NewCPUEngine()
	}

	fractalCortex := NewFractalCortex(cfg, engine)
	if err := fractalCortex.Load(cfg.DataDir); err != nil {
		fmt.Printf("[FractalCortex] No saved weights found or load failed (%v), starting fresh.\n", err)
	} else {
		fmt.Println("[FractalCortex] Successfully restored BitNet weights from disk.")
	}

	// 13. RadioCortex — load from disk, fall back to fresh if missing.
	var radioCortex *RadioCortex
	var signalCodec *SignalCodec
	radioPath := filepath.Join(cfg.DataDir, "radio_cortex.nxrc")
	if _, err := os.Stat(radioPath); err == nil {
		rc, sc, loadErr := LoadRadioCortex(radioPath)
		if loadErr != nil {
			fmt.Printf("[RadioCortex] Load failed (%v), starting fresh.\n", loadErr)
			radioCortex = NewRadioCortex(cfg.RadioNeuronCount, rng)
			signalCodec = NewSignalCodec(1000)
		} else {
			radioCortex = rc
			if sc != nil {
				signalCodec = sc
			} else {
				signalCodec = NewSignalCodec(1000)
			}
			fmt.Printf("[RadioCortex] Restored %d neurons (tick %d) from disk.\n", rc.Size, rc.TickCount)
		}
	} else {
		radioCortex = NewRadioCortex(cfg.RadioNeuronCount, rng)
		signalCodec = NewSignalCodec(1000)
	}

	// 8. Broca 2.0 — Load BPE tokenizer and MiniTransformer (optional)
	var bpeTokenizer *BPETokenizer
	var miniTransformer *MiniTransformer

	tokenizerPath := filepath.Join(cfg.DataDir, "tokenizer.json")
	if loadedTok, err := LoadBPETokenizer(tokenizerPath); err == nil {
		bpeTokenizer = loadedTok
		fmt.Printf("[Broca 2.0] BPE tokenizer loaded (vocab: %d, merges: %d)\n",
			loadedTok.ActualVocabSize(), len(loadedTok.Merges))

		// Create transformer matching tokenizer vocab
		tfCfg := DefaultTransformerConfig(loadedTok.ActualVocabSize())
		miniTransformer = NewMiniTransformer(tfCfg, rng)
		fmt.Printf("[Broca 2.0] MiniTransformer initialized (%d params)\n",
			miniTransformer.ParamCount())
	} else {
		fmt.Println("[Broca 2.0] No tokenizer found, using Broca 1.0 fallback.")
	}

	o := &Organism{
		Config:  cfg,
		Vocab:   vocab,
		Encoder: encoder,
		Decoder: decoder,
		Brain:   brain,

		Wernicke:       wernicke,
		Broca:          NewBroca(vocab, encoder, decoder, brain, cfg),
		Hippocampus:    hippocampus,
		SemanticMemory: semanticMemory,
		Cerebellum:     NewCerebellum(cfg), // Volatile — rebuilt from experience
		Prefrontal:     prefrontal,

		Predictor:         NewPredictor(rng),
		Workspace:         NewWorkspace(cfg),
		ThousandBrains:    NewThousandBrains(cfg, rng),
		Reward:            reward,
		Emotion:           emotion,
		Self:              self,
		Curiosity:         curiosity,
		WorkingMem:        NewWorkingMemory(cfg),
		Attention:         NewAttentionModule(cfg),
		ErrorLearner:      NewErrorDrivenLearner(cfg),
		Analogy:           NewAnalogyEngine(cfg, brain, encoder),
		SleepConsolidator: NewSleepConsolidator(cfg),
		Reasoning:         NewReasoningEngine(),

		// FractalCortex: Infinite growing MoC ALBERT layers
		FractalCortex: fractalCortex,

		// RadioCortex: frequency-based neural processor (restored or fresh)
		RadioCortex: radioCortex,
		SignalCodec: signalCodec,

		// Broca 2.0: Transformer language model (nil if no tokenizer on disk)
		Transformer: miniTransformer,
		Tokenizer:   bpeTokenizer,

		Sensory: NewSensorySystem(encoder, cfg),
		Motor:   NewMotorSystem(cfg),
		Rhythm:  NewRhythmEngine(cfg),

		InteractionCount: interactionCount,
		Rng:              rng,
	}

	return o, nil
}

// HandleFeedback applies positive or negative human reinforcement to the organism's synapses, emotional state, and drives.
func (o *Organism) HandleFeedback(topic string, responseText string, positive bool, correctText string) {
	if positive {
		o.Self.RecordSuccess(topic, 255)

		// Modulate Emotion: boost valence and arousal (rewarding)
		o.Emotion.Update(100, 0, atomic.LoadUint64(&o.InteractionCount))

		// Reinforce the response sequence in Brain and SequenceMemory
		if responseText != "" {
			o.Brain.ReinforceSequence(responseText, true)
		}
	} else {
		o.Self.RecordFailure(topic)

		// Modulate Emotion: drop valence, raise arousal (frustrated/alert)
		o.Emotion.Update(-100, 100, atomic.LoadUint64(&o.InteractionCount))

		// Weaken the incorrect sequence in Brain and SequenceMemory
		if responseText != "" {
			o.Brain.ReinforceSequence(responseText, false)
		}

		// Evict the incorrect mapping from Cerebellum cache
		if o.Cerebellum != nil && responseText != "" {
			o.Cerebellum.EvictByResponse(responseText)
		}

		// Learn correct text if provided
		if correctText != "" {
			o.Learn(correctText)
		}
	}

	if !o.Config.NoSave {
		_ = o.Save(o.Config.DataDir)
	}
}

// SelfTrain performs active self-reflection cycles on the organism's knowledge.
// It retrieves key concepts, deliberates alternative phrasings internally,
// checks their spiking stability in the Prefrontal reservoir, and strengthens or prunes synapses.
func (o *Organism) SelfTrain() (consolidated int, pruned int, logMsgs []string) {
	logMsgs = append(logMsgs, "Starting autonomous prefrontal self-reflection loop...")

	// Get strong topics to reflect upon
	topics := o.Self.StrongTopics()
	if len(topics) == 0 {
		// Fallback: use all known topics if no strong topics yet
		for topic := range o.Self.KnownTopics {
			topics = append(topics, topic)
		}
	}
	if len(topics) == 0 {
		logMsgs = append(logMsgs, "Self-reflection idle: no topics known yet.")
		return 0, 0, logMsgs
	}

	// Limit to a max of N topics per training run to prevent long pauses
	maxTopics := o.Config.SelfTrainMaxTopics
	if maxTopics <= 0 {
		maxTopics = 5
	}
	if len(topics) > maxTopics {
		// Shuffle or select randomly
		o.Rng.Shuffle(len(topics), func(i, j int) {
			topics[i], topics[j] = topics[j], topics[i]
		})
		topics = topics[:maxTopics]
	}

	for _, topic := range topics {
		if topic == "" {
			continue
		}
		logMsgs = append(logMsgs, fmt.Sprintf("Reflecting on concept: '%s'...", topic))

		// Find semantic memory prototypes or episodic memory matches for this topic
		sdr := o.Encoder.EncodeSentence(topic)
		recallThresh := o.Config.SelfTrainRecallThresh
		if recallThresh == 0 {
			recallThresh = 50
		}
		mem, found := o.Hippocampus.Recall(sdr, recallThresh)
		if !found {
			continue
		}

		// Let Broca generate a phrasing representing this concept
		phrase := o.Broca.Generate(mem.Output, o.Config.MaxGenWords)
		if phrase == "" || phrase == "(no confident response)" {
			continue
		}

		// Simulate this phrase's SDR in Prefrontal reasoning reservoir
		phraseSDR := o.Encoder.EncodeSentence(phrase)
		o.Prefrontal.Think(phraseSDR, o.Config.ThinkCycles)
		conf := o.Prefrontal.GetConfidence()

		logMsgs = append(logMsgs, fmt.Sprintf("  Internal phrasing: '%s' (Prefrontal Stability: %d/255)", phrase, conf))

		stableConf := o.Config.SelfTrainStableConf
		if stableConf == 0 {
			stableConf = 200
		}
		unstableConf := o.Config.SelfTrainUnstableConf
		if unstableConf == 0 {
			unstableConf = 100
		}

		if conf > stableConf {
			// Attractor is stable! Reinforce the synapses
			o.Brain.ReinforceSequence(phrase, true)
			o.Self.RecordSuccess(topic, conf)
			consolidated++
			logMsgs = append(logMsgs, "  --> Attractor matches stable prefrontal state. Synapses reinforced.")
		} else if conf < unstableConf {
			// Unstable attractor. Prune it!
			o.Brain.ReinforceSequence(phrase, false)
			o.Self.RecordFailure(topic)
			pruned++
			logMsgs = append(logMsgs, "  --> Unstable network firing detected. Pruning obsolete pathways.")
		} else {
			logMsgs = append(logMsgs, "  --> Marginally stable. Retained in working memory without changes.")
		}
	}

	return consolidated, pruned, logMsgs
}
