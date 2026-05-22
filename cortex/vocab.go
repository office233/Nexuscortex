package cortex

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"unicode"
)

// ─────────────────────────────────────────────────────────────────────
// Vocabulary — Maps words to neuron IDs and back
// ─────────────────────────────────────────────────────────────────────
//
// To generate text like an LLM, we need word-level neurons,
// not just character-level. Each unique word gets a uint16 ID
// (supporting up to 65,535 words).

// Vocab manages the bidirectional word↔ID mapping.
type Vocab struct {
	WordToID map[string]uint32 `json:"word_to_id"`
	IDToWord map[uint32]string `json:"id_to_word"`
	NextID   uint32            `json:"next_id"`

	mu sync.RWMutex
}

// NewVocab creates an empty vocabulary.
func NewVocab() *Vocab {
	return &Vocab{
		WordToID: make(map[string]uint32),
		IDToWord: make(map[uint32]string),
		NextID:   1, // 0 is reserved for <UNK>
	}
}

// GetOrCreate returns the ID for a word, creating it if new.
// GetOrCreateChecked returns the ID for a word, or an error if vocabulary is full.
func (v *Vocab) GetOrCreateChecked(word string) (uint32, error) {
	word = strings.ToLower(strings.TrimSpace(word))
	if word == "" {
		return 0, nil
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	if id, ok := v.WordToID[word]; ok {
		return id, nil
	}

	if v.NextID == 4294967295 {
		return 0, fmt.Errorf("vocabulary full (limit 4294967295)")
	}

	id := v.NextID
	v.WordToID[word] = id
	v.IDToWord[id] = word
	v.NextID++
	return id, nil
}

// GetOrCreate returns the ID for a word, creating it if new.
// Compatibility wrapper around GetOrCreateChecked.
func (v *Vocab) GetOrCreate(word string) uint32 {
	id, _ := v.GetOrCreateChecked(word)
	return id
}

// Get returns the ID for a word (0 if unknown).
func (v *Vocab) Get(word string) uint32 {
	word = strings.ToLower(strings.TrimSpace(word))
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.WordToID[word]
}

// Decode returns the word for an ID.
func (v *Vocab) Decode(id uint32) string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	if w, ok := v.IDToWord[id]; ok {
		return w
	}
	return "<UNK>"
}

// Size returns the number of words in the vocabulary.
func (v *Vocab) Size() int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return len(v.WordToID)
}

// Save writes the vocabulary to a JSON file.
func (v *Vocab) Save(path string) error {
	v.mu.RLock()
	defer v.mu.RUnlock()

	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal vocab: %w", err)
	}
	return os.WriteFile(path, data, 0600)
}

// LoadVocab reads a vocabulary from a JSON file.
func LoadVocab(path string) (*Vocab, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}

	v := NewVocab()
	if err := json.Unmarshal(data, v); err != nil {
		return nil, fmt.Errorf("unmarshal vocab: %w", err)
	}
	return v, nil
}

// Tokenize splits text into word tokens (simple whitespace + punctuation split).
// Unicode-aware: preserves diacritics and non-Latin scripts.
func Tokenize(text string) []string {
	// Split punctuation from words so "hello," becomes ["hello", ","]
	var tokens []string
	current := strings.Builder{}

	for _, r := range text {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '\'', r == '-':
			current.WriteRune(r)
		default:
			if current.Len() > 0 {
				tokens = append(tokens, strings.ToLower(current.String()))
				current.Reset()
			}
			if r != ' ' && r != '\t' {
				tokens = append(tokens, string(r))
			}
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, strings.ToLower(current.String()))
	}
	return tokens
}
