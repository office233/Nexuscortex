package cortex

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// ─────────────────────────────────────────────────────────────────────
// Brain — Word-Level Self-Growing Neural Network
// ─────────────────────────────────────────────────────────────────────
//
// The Brain is the V2 cortex engine that operates at the WORD level.
// Each synapse connects two word-neurons.
//
// To support uint32 Source and Target while maintaining the biologically-
// plausible RGBA32 format, the binary representation uses 12 bytes
// (exactly 3 pixels of RGBA32):
//
//   Synapse Sequence (3 × RGBA32 pixels):
//     Pixel 1: [Source_B3, Source_B2, Source_B1, Source_B0]  (4-byte uint32)
//     Pixel 2: [Target_B3, Target_B2, Target_B1, Target_B0]  (4-byte uint32)
//     Pixel 3: [Weight_Hi, Weight_Lo, Flags, Age]             (4-byte weight/meta)
//
// This gives us:
//   - 4,294,967,295 possible word neurons (uint32 source/target)
//   - 65,535 weight resolution (uint16)
//   - 256 flag combinations
//   - 256 age levels

// Synapse represents a weighted connection to a target word-neuron.
type Synapse struct {
	Target uint32 // Target word neuron ID
	Weight uint16 // Connection strength (higher = stronger association)
	Flags  uint8  // Synapse type and flags
	Age    uint8  // How many cycles since last activation
}

// Synapse flags.
const (
	SynFlagActive     uint8 = 0x01
	SynFlagSequential uint8 = 0x02 // Word A is followed by word B
	SynFlagSemantic   uint8 = 0x04 // Words appear in same context
	SynFlagSkipGram   uint8 = 0x08 // Words are 2+ positions apart
)

// DefaultProceduralSynapseCount is the number of deterministic baseline
// synapses generated per word in GetProceduralSynapses.
const DefaultProceduralSynapseCount = 5

// Brain is the word-level self-growing neural network.
type Brain struct {
	Vocab     *Vocab
	Config    Config
	Synapses  map[uint32][]Synapse // SourceID -> TargetSynapses
	FilePath  string
	VocabPath string
	SeqPath   string
	Rng       *rand.Rand
	SeqMem    *SequenceMemory

	mu sync.RWMutex
}

// NewBrain creates a new empty brain.
func NewBrain(filePath, vocabPath string, rng *rand.Rand, cfg Config) *Brain {
	seqPath := filepath.Join(filepath.Dir(filePath), "sequence_memory.json")
	return &Brain{
		Vocab:     NewVocab(),
		Config:    cfg,
		Synapses:  make(map[uint32][]Synapse),
		FilePath:  filePath,
		VocabPath: vocabPath,
		SeqPath:   seqPath,
		Rng:       rng,
		SeqMem:    NewSequenceMemory(cfg),
	}
}

// ─────────────────────────────────────────────────────────────────────
// PROCEDURAL & SEEDED SYNAPSES
// ─────────────────────────────────────────────────────────────────────

// GetProceduralSynapses generates deterministic synapses for a sourceID using LCG spatial hashing.
func (b *Brain) GetProceduralSynapses(sourceID uint32) []Synapse {
	if len(b.Synapses) == 0 {
		return nil
	}
	vSize := b.Vocab.Size()
	if vSize <= 1 {
		return nil
	}

	seed := uint64(sourceID)
	numSynapses := DefaultProceduralSynapseCount
	syns := make([]Synapse, 0, numSynapses)

	for i := 0; i < numSynapses; i++ {
		// LCG step
		seed = seed*2862933555777941757 + 3037000493
		nextID := b.Vocab.NextID
		if nextID <= 1 {
			continue
		}

		target := uint32((seed % uint64(nextID-1)) + 1)
		if target == sourceID {
			continue
		}

		weight := uint16(5 + (seed % 21)) // baseline weight in [5, 25]

		syns = append(syns, Synapse{
			Target: target,
			Weight: weight,
			Flags:  SynFlagActive | SynFlagSequential,
			Age:    0,
		})
	}
	return syns
}

// GetMergedSynapses merges dynamic learned synapses with procedural baseline synapses.
func (b *Brain) GetMergedSynapses(sourceID uint32) []Synapse {
	plastic := b.Synapses[sourceID]
	procedural := b.GetProceduralSynapses(sourceID)

	merged := make([]Synapse, 0, len(plastic)+len(procedural))
	merged = append(merged, plastic...)

	for _, ps := range procedural {
		found := false
		for j := range merged {
			if merged[j].Target == ps.Target {
				newW := uint32(merged[j].Weight) + uint32(ps.Weight)
				if newW > 65535 {
					newW = 65535
				}
				merged[j].Weight = uint16(newW)
				merged[j].Flags |= ps.Flags
				found = true
				break
			}
		}
		if !found {
			merged = append(merged, ps)
		}
	}

	return merged
}

// ─────────────────────────────────────────────────────────────────────
// LEARN — The brain teaches itself from text
// ─────────────────────────────────────────────────────────────────────

// Learn feeds text into the brain. It creates word-neurons and
// associations between consecutive words (bigrams) and nearby words
// (skip-grams). This is how the brain grows.
//
// Returns the number of new synapses created.
func (b *Brain) Learn(text string) int {
	tokens := Tokenize(text)
	if len(tokens) < 2 {
		return 0
	}

	// Convert words to neuron IDs (creating new neurons as needed).
	ids := make([]uint32, len(tokens))
	for i, token := range tokens {
		ids[i] = b.Vocab.GetOrCreate(token)
	}

	// Feed to sequence memory.
	b.mu.Lock()
	defer b.mu.Unlock()

	// Feed to sequence memory (must be inside lock to prevent data race).
	if b.SeqMem != nil {
		b.SeqMem.Learn(ids)
	}

	added := 0

	// Bigrams: word[i] → word[i+1] (sequential association).
	for i := 0; i < len(ids)-1; i++ {
		if b.strengthenOrCreate(ids[i], ids[i+1], b.Config.BrainBigramWeight, SynFlagActive|SynFlagSequential) {
			added++
		}
	}

	// Skip-1-grams: word[i] → word[i+2] (context association).
	for i := 0; i < len(ids)-2; i++ {
		if b.strengthenOrCreate(ids[i], ids[i+2], b.Config.BrainSkipgramWeight, SynFlagActive|SynFlagSkipGram) {
			added++
		}
	}

	// Co-occurrence in window: semantic association.
	windowSize := b.Config.BrainContextWindowSize
	if windowSize <= 0 {
		windowSize = 5
	}
	for i := 0; i < len(ids); i++ {
		for j := i + 2; j < len(ids) && j < i+windowSize; j++ {
			if ids[i] != ids[j] {
				if b.strengthenOrCreate(ids[i], ids[j], b.Config.BrainSemanticWeight, SynFlagActive|SynFlagSemantic) {
					added++
				}
			}
		}
	}

	return added
}

// LearnBatch feeds multiple texts into the brain.
func (b *Brain) LearnBatch(texts []string) int {
	total := 0
	for _, text := range texts {
		total += b.Learn(text)
	}
	return total
}

// ─────────────────────────────────────────────────────────────────────
// GENERATE — The brain writes by itself!
// ─────────────────────────────────────────────────────────────────────

// Generate produces text by following the strongest association chains.
// Given a prompt, it predicts the next word by finding the target
// with the highest weight from the current word, then repeats.
//
// This is the core "LLM-like" behavior: next-token prediction
// using associative memory instead of matrix multiplication.
func (b *Brain) Generate(prompt string, maxWords int) string {
	tokens := Tokenize(prompt)
	if len(tokens) == 0 {
		return ""
	}

	b.mu.RLock()
	defer b.mu.RUnlock()

	// Start from the last word in the prompt.
	output := make([]string, 0, maxWords)

	currentID := b.Vocab.Get(tokens[len(tokens)-1])
	if currentID == 0 {
		// Unknown word — try the second-to-last.
		if len(tokens) >= 2 {
			currentID = b.Vocab.Get(tokens[len(tokens)-2])
		}
		if currentID == 0 {
			// Nothing learned about these words yet. Return empty.
			return ""
		}
	}

	// Track recently used words to avoid loops.
	recentlyUsed := make(map[uint32]int)
	for _, t := range tokens {
		recentlyUsed[b.Vocab.Get(t)] = 1
	}

	// Build the context sequence of word IDs for sequence memory
	contextIDs := make([]uint32, 0, len(tokens)+maxWords)
	for _, t := range tokens {
		id := b.Vocab.Get(t)
		if id != 0 {
			contextIDs = append(contextIDs, id)
		}
	}

	for i := 0; i < maxWords; i++ {
		var nextID uint32
		var found bool

		// 1. Try multi-context SequenceMemory first
		if b.SeqMem != nil && len(contextIDs) > 0 {
			nextID, found = b.SeqMem.Predict(contextIDs, recentlyUsed)
		}

		// 2. Fall back to bigram associative synapses
		if !found {
			nextID, found = b.predictNext(currentID, recentlyUsed)
		}

		if !found {
			break
		}

		word := b.Vocab.Decode(nextID)
		output = append(output, word)
		contextIDs = append(contextIDs, nextID)

		// Stop at punctuation — but the Brain learns what punctuation IS
		// from the text it's fed. If a word has no outgoing synapses,
		// predictNext returns false and we stop naturally.
		// Punctuation tokens naturally have few outgoing connections,
		// so stopping is emergent, not hardcoded.

		recentlyUsed[nextID]++
		currentID = nextID
	}

	return b.formatOutput(output)
}

// GenerateMultiple generates multiple candidate responses and returns
// the best one (longest coherent chain). This simulates beam search.
func (b *Brain) GenerateMultiple(prompt string, maxWords, candidates int) string {
	best := ""
	bestLen := 0
	for i := 0; i < candidates; i++ {
		candidate := b.Generate(prompt, maxWords)
		words := len(strings.Fields(candidate))
		if words > bestLen {
			best = candidate
			bestLen = words
		}
	}
	return best
}

// predictNext finds the most likely next word given the current word.
// Uses weighted random selection to add variety.
func (b *Brain) predictNext(currentID uint32, recentlyUsed map[uint32]int) (uint32, bool) {
	type candidate struct {
		id     uint32
		weight uint16
	}

	synapses := b.GetMergedSynapses(currentID)
	if len(synapses) == 0 {
		return 0, false
	}

	var candidates []candidate
	for _, syn := range synapses {
		if syn.Flags&SynFlagActive == 0 || syn.Weight == 0 {
			continue
		}

		// Sequential synapses are strongly preferred for generation.
		// Use uint32 to prevent overflow when multiplying.
		score := uint32(syn.Weight)
		if syn.Flags&SynFlagSequential != 0 {
			mult := uint32(b.Config.BrocaSequentialMultiplier)
			if mult == 0 {
				mult = 3 // Fallback default
			}
			score *= mult
			if score > 65535 {
				score = 65535
			}
		}
		w := uint16(score)

		// Penalize recently used words (anti-loop).
		if count, ok := recentlyUsed[syn.Target]; ok && count > 0 {
			w /= uint16(count + 1)
		}

		if w > 0 {
			candidates = append(candidates, candidate{syn.Target, w})
		}
	}

	if len(candidates) == 0 {
		return 0, false
	}

	// Sort by weight descending.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].weight > candidates[j].weight
	})

	// Weighted random selection among top candidates (temperature sampling).
	topN := len(candidates)
	limit := b.Config.BrocaTopNCandidates
	if limit <= 0 {
		limit = 5 // Fallback default
	}
	if topN > limit {
		topN = limit
	}
	top := candidates[:topN]

	totalWeight := uint32(0)
	for _, c := range top {
		totalWeight += uint32(c.weight)
	}

	if totalWeight == 0 {
		return top[0].id, true
	}

	roll := uint32(b.Rng.Intn(int(totalWeight)))
	cumulative := uint32(0)
	for _, c := range top {
		cumulative += uint32(c.weight)
		if roll < cumulative {
			return c.id, true
		}
	}

	return top[0].id, true
}

// formatOutput joins tokens with proper spacing around punctuation.
func (b *Brain) formatOutput(tokens []string) string {
	if len(tokens) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(tokens[0])

	for i := 1; i < len(tokens); i++ {
		t := tokens[i]
		if len(t) == 1 {
			r := rune(t[0])
			if r == '.' || r == ',' || r == '!' || r == '?' || r == ':' || r == ';' {
				sb.WriteString(t)
				continue
			}
		}
		sb.WriteByte(' ')
		sb.WriteString(t)
	}

	return sb.String()
}

// ─────────────────────────────────────────────────────────────────────
// PRUNE — Forget old, unused connections
// ─────────────────────────────────────────────────────────────────────

// Prune removes synapses older than maxAge. Returns count removed.
func (b *Brain) Prune(maxAge uint8) int {
	b.mu.Lock()
	defer b.mu.Unlock()

	removed := 0
	for srcID, syns := range b.Synapses {
		alive := make([]Synapse, 0, len(syns))
		for i := range syns {
			s := &syns[i]
			if s.Flags&SynFlagActive == 0 || s.Age > maxAge {
				removed++
				continue
			}
			s.Age++
			alive = append(alive, *s)
		}
		if len(alive) == 0 {
			delete(b.Synapses, srcID)
		} else {
			b.Synapses[srcID] = alive
		}
	}
	return removed
}

// ─────────────────────────────────────────────────────────────────────
// PERSIST — Save and load the brain
// ─────────────────────────────────────────────────────────────────────

// BrainFileMagic identifies a .brain file.
const BrainFileMagic = "NXBRAIN1"

// Save writes the brain to disk (synapses as binary + vocab as JSON).
func (b *Brain) Save() error {
	b.mu.RLock()
	defer b.mu.RUnlock()

	// Save vocab.
	if err := b.Vocab.Save(b.VocabPath); err != nil {
		return fmt.Errorf("save vocab: %w", err)
	}

	// Save sequence memory.
	if b.SeqMem != nil {
		if err := b.SeqMem.Save(b.SeqPath); err != nil {
			return fmt.Errorf("save sequence memory: %w", err)
		}
	}

	// Save synapses as binary (atomic: temp file → sync → rename).
	tmpPath := b.FilePath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create brain file: %w", err)
	}

	// Write magic.
	if _, err := f.Write([]byte(BrainFileMagic)); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return err
	}

	// Count total synapses
	totalSynapses := 0
	for _, syns := range b.Synapses {
		totalSynapses += len(syns)
	}

	// Write synapse count.
	if err := binary.Write(f, binary.LittleEndian, uint32(totalSynapses)); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return err
	}

	// Write each synapse as 12 bytes (3 × RGBA32).
	for srcID, syns := range b.Synapses {
		for _, s := range syns {
			buf := [12]byte{
				byte(srcID >> 24), byte(srcID >> 16), byte(srcID >> 8), byte(srcID), // Pixel 1: Source
				byte(s.Target >> 24), byte(s.Target >> 16), byte(s.Target >> 8), byte(s.Target), // Pixel 2: Target
				byte(s.Weight >> 8), byte(s.Weight), // Pixel 3: Weight
				s.Flags,                             // Pixel 3: Flags
				s.Age,                               // Pixel 3: Age
			}
			if _, err := f.Write(buf[:]); err != nil {
				f.Close()
				os.Remove(tmpPath)
				return err
			}
		}
	}

	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("sync brain file: %w", err)
	}
	f.Close()

	if err := os.Rename(tmpPath, b.FilePath); err != nil {
		return fmt.Errorf("rename brain file: %w", err)
	}
	return nil
}

// LoadBrain reads a brain from disk.
func LoadBrain(filePath, vocabPath string, rng *rand.Rand, cfg Config) (*Brain, error) {
	vocab, err := LoadVocab(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("load vocab: %w", err)
	}

	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("open brain file: %w", err)
	}
	defer f.Close()

	// Read magic.
	magic := make([]byte, 8)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != BrainFileMagic {
		return nil, fmt.Errorf("invalid magic: %q", string(magic))
	}

	// Read synapse count.
	var count uint32
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read count: %w", err)
	}

	// Safety: reject absurdly large counts to prevent OOM from corrupted files.
	const MaxSynapseCount uint32 = 10_000_000
	if count > MaxSynapseCount {
		return nil, fmt.Errorf("synapse count %d exceeds safety limit %d", count, MaxSynapseCount)
	}

	synapses := make(map[uint32][]Synapse)
	for i := uint32(0); i < count; i++ {
		var buf [12]byte
		if _, err := io.ReadFull(f, buf[:]); err != nil {
			return nil, fmt.Errorf("read synapse %d: %w", i, err)
		}
		srcID := uint32(buf[0])<<24 | uint32(buf[1])<<16 | uint32(buf[2])<<8 | uint32(buf[3])
		targetID := uint32(buf[4])<<24 | uint32(buf[5])<<16 | uint32(buf[6])<<8 | uint32(buf[7])
		weight := uint16(buf[8])<<8 | uint16(buf[9])
		flags := buf[10]
		age := buf[11]

		synapses[srcID] = append(synapses[srcID], Synapse{
			Target: targetID,
			Weight: weight,
			Flags:  flags,
			Age:    age,
		})
	}

	// Load sequence memory.
	seqPath := filepath.Join(filepath.Dir(filePath), "sequence_memory.json")
	seqMem, err := LoadSequenceMemory(seqPath, cfg)
	if err != nil {
		return nil, fmt.Errorf("load sequence memory: %w", err)
	}
	if seqMem == nil {
		seqMem = NewSequenceMemory(cfg)
	}

	return &Brain{
		Vocab:     vocab,
		Config:    cfg,
		Synapses:  synapses,
		FilePath:  filePath,
		VocabPath: vocabPath,
		SeqPath:   seqPath,
		Rng:       rng,
		SeqMem:    seqMem,
	}, nil
}

// ─────────────────────────────────────────────────────────────────────
// STATS
// ─────────────────────────────────────────────────────────────────────

// BrainStats holds brain statistics.
type BrainStats struct {
	VocabSize       int
	TotalSynapses   int
	ActiveSynapses  int
	SequentialCount int
	SemanticCount   int
	MemoryBytes     int
	AvgWeight       uint32  // ×256 for precision, like NetworkStats
}

// GetStats returns brain statistics.
func (b *Brain) GetStats() BrainStats {
	b.mu.RLock()
	defer b.mu.RUnlock()

	totalSynapses := 0
	for _, syns := range b.Synapses {
		totalSynapses += len(syns)
	}

	s := BrainStats{
		VocabSize:     b.Vocab.Size(),
		TotalSynapses: totalSynapses,
		MemoryBytes:   totalSynapses * 12, // 12 bytes per synapse
	}

	var totalW uint64
	for _, syns := range b.Synapses {
		for _, syn := range syns {
			if syn.Flags&SynFlagActive != 0 {
				s.ActiveSynapses++
				totalW += uint64(syn.Weight)
			}
			if syn.Flags&SynFlagSequential != 0 {
				s.SequentialCount++
			}
			if syn.Flags&SynFlagSemantic != 0 {
				s.SemanticCount++
			}
		}
	}
	if s.ActiveSynapses > 0 {
		s.AvgWeight = uint32((totalW * 256) / uint64(s.ActiveSynapses))
	}
	return s
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers (must be called with lock held)
// ─────────────────────────────────────────────────────────────────────

// strengthenOrCreate either strengthens an existing synapse or creates
// a new one. Returns true if a NEW synapse was created.
func (b *Brain) strengthenOrCreate(source, target uint32, amount uint16, flags uint8) bool {
	syns := b.Synapses[source]
	for i := range syns {
		s := &syns[i]
		if s.Target == target {
			// Strengthen existing synapse.
			newW := uint32(s.Weight) + uint32(amount)
			if newW > 65535 {
				newW = 65535
			}
			s.Weight = uint16(newW)
			s.Age = 0 // Reset age — this is being used!
			s.Flags |= flags
			return false
		}
	}

	// Create new synapse.
	b.Synapses[source] = append(syns, Synapse{
		Target: target,
		Weight: amount,
		Flags:  flags,
		Age:    0,
	})
	return true
}

// ReinforceSequence strengthens or weakens a sequential chain of words.
func (b *Brain) ReinforceSequence(text string, positive bool) {
	tokens := Tokenize(text)
	if len(tokens) < 2 {
		return
	}

	// Convert words to neuron IDs
	ids := make([]uint32, len(tokens))
	b.mu.Lock()
	defer b.mu.Unlock()
	for i, token := range tokens {
		ids[i] = b.Vocab.GetOrCreate(token)
	}

	// Adjust bigram transition weights
	ltpAmt := b.Config.FeedbackLTPAmount
	if ltpAmt == 0 {
		ltpAmt = 50
	}
	ltdAmt := b.Config.FeedbackLTDAmount
	if ltdAmt == 0 {
		ltdAmt = 40
	}
	newSynW := b.Config.FeedbackNewSynapseWeight
	if newSynW == 0 {
		newSynW = 50
	}
	for i := 0; i < len(ids)-1; i++ {
		source := ids[i]
		target := ids[i+1]
		found := false
		syns := b.Synapses[source]
		for j := range syns {
			s := &syns[j]
			if s.Target == target {
				found = true
				if positive {
					// LTP: strengthen
					newW := uint32(s.Weight) + uint32(ltpAmt)
					if newW > 65535 {
						newW = 65535
					}
					s.Weight = uint16(newW)
					s.Age = 0
				} else {
					// LTD: weaken
					if s.Weight > ltdAmt {
						s.Weight -= ltdAmt
					} else {
						s.Weight = 0
						s.Flags &= ^SynFlagActive // Deactivate
					}
				}
				break
			}
		}
		if !found && positive {
			// Create it if positive feedback and missing
			b.Synapses[source] = append(syns, Synapse{
				Target: target,
				Weight: newSynW,
				Flags:  SynFlagActive | SynFlagSequential,
				Age:    0,
			})
		}
	}

	// Adjust multi-context sequence memory transitions (inside lock)
	if b.SeqMem != nil {
		b.SeqMem.ReinforceSequence(ids, positive)
	}
}


