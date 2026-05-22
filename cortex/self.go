package cortex

import (
	"encoding/json"
	"fmt"
	"os"
)

// ─────────────────────────────────────────────────────────────────────
// self.go — Self-Model (Meta-cognition)
// ─────────────────────────────────────────────────────────────────────
//
// The self-model tracks what the organism knows and doesn't know.
// This is NOT personality — it's awareness of own competence.
//
// Each topic maintains a competence level (0–255) that is updated
// every time the organism records a success or failure on that topic.
// Success reinforces competence; failure erodes it. The model also
// tracks aggregate accuracy as a ratio of successes to total inputs,
// mapped to the 0–255 range.
//
// This gives the organism a lightweight introspective signal it can
// use to decide whether to attempt an answer, defer, or seek help.

// SelfModel tracks the organism's awareness of its own competence
// across topics, plus aggregate accuracy statistics.
type SelfModel struct {
	KnownTopics   map[string]uint8 // topic -> competence level (0-255)
	SuccessCount  uint64           // How many times output was confident
	FailureCount  uint64           // How many times output was uncertain
	TotalInputs   uint64           // Total inputs processed
	AvgConfidence uint8            // Running average confidence
	Cfg           Config           // Configuration for thresholds
}

// NewSelfModel creates an empty self-model with no topic knowledge.
func NewSelfModel(cfgs ...Config) *SelfModel {
	var cfg Config
	if len(cfgs) > 0 {
		cfg = cfgs[0]
	} else {
		cfg = DefaultConfig()
	}
	return &SelfModel{
		KnownTopics: make(map[string]uint8),
		Cfg:         cfg,
	}
}

// RecordSuccess records a confident, successful output on the given
// topic. The topic's competence is moved toward the reported
// confidence using an exponential moving average (EMA) with α ≈ 1/4.
// The global running average confidence is updated the same way.
func (s *SelfModel) RecordSuccess(topic string, confidence uint8) {
	s.SuccessCount++
	s.TotalInputs++

	// Update topic competence: EMA toward reported confidence.
	prev := s.KnownTopics[topic] // 0 if new topic
	s.KnownTopics[topic] = uint8((uint16(prev)*3 + uint16(confidence)) / 4)

	// Update running average confidence.
	s.AvgConfidence = uint8((uint16(s.AvgConfidence)*3 + uint16(confidence)) / 4)
}

// RecordFailure records an uncertain or failed output on the given
// topic. Competence for the topic is decayed by 25%.
func (s *SelfModel) RecordFailure(topic string) {
	s.FailureCount++
	s.TotalInputs++

	prev := s.KnownTopics[topic]
	// Decay competence by 25%.
	s.KnownTopics[topic] = uint8((uint16(prev) * 3) / 4)

	// Drag running confidence toward 0.
	s.AvgConfidence = uint8((uint16(s.AvgConfidence) * 3) / 4)
}

// GetCompetence returns the competence level for a topic (0–255).
// Returns 0 for unknown topics.
func (s *SelfModel) GetCompetence(topic string) uint8 {
	return s.KnownTopics[topic]
}

// KnowsAbout returns true if the organism has non-trivial competence
// on the topic (competence > SelfKnowsAboutThreshold from Config).
func (s *SelfModel) KnowsAbout(topic string) bool {
	thresh := s.Cfg.SelfKnowsAboutThreshold
	if thresh == 0 {
		thresh = 50
	}
	return uint32(s.KnownTopics[topic]) > thresh
}

// WeakTopics returns all topics where competence is below SelfWeakTopicThreshold.
// Only topics that have been observed at least once are considered.
func (s *SelfModel) WeakTopics() []string {
	thresh := s.Cfg.SelfWeakTopicThreshold
	if thresh == 0 {
		thresh = 30
	}
	var result []string
	for topic, comp := range s.KnownTopics {
		if uint32(comp) < thresh {
			result = append(result, topic)
		}
	}
	return result
}

// StrongTopics returns all topics where competence exceeds SelfStrongTopicThreshold.
func (s *SelfModel) StrongTopics() []string {
	thresh := s.Cfg.SelfStrongTopicThreshold
	if thresh == 0 {
		thresh = 200
	}
	var result []string
	for topic, comp := range s.KnownTopics {
		if uint32(comp) > thresh {
			result = append(result, topic)
		}
	}
	return result
}

// GetAccuracy returns the organism's overall accuracy as a 0–255
// value, computed as (SuccessCount / TotalInputs) × 255.
// Returns 0 if no inputs have been processed.
// M1 fix: guards against uint64 overflow when SuccessCount is very large.
func (s *SelfModel) GetAccuracy() uint8 {
	if s.TotalInputs == 0 {
		return 0
	}
	// Overflow guard: if SuccessCount * 255 would overflow uint64,
	// SuccessCount must be >= TotalInputs, so accuracy is 255.
	const maxSafe = ^uint64(0) / 255
	if s.SuccessCount > maxSafe {
		return 255
	}
	return uint8((s.SuccessCount * 255) / s.TotalInputs)
}

// Reset clears all competence data and resets counters to zero.
func (s *SelfModel) Reset() {
	s.KnownTopics = make(map[string]uint8)
	s.SuccessCount = 0
	s.FailureCount = 0
	s.TotalInputs = 0
	s.AvgConfidence = 0
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save / Load
// ─────────────────────────────────────────────────────────────────────

// selfSaveData is the JSON-serializable snapshot of SelfModel state.
type selfSaveData struct {
	KnownTopics   map[string]uint8 `json:"known_topics"`
	SuccessCount  uint64           `json:"success_count"`
	FailureCount  uint64           `json:"failure_count"`
	TotalInputs   uint64           `json:"total_inputs"`
	AvgConfidence uint8            `json:"avg_confidence"`
}

// Save persists the SelfModel state to a JSON file.
func (s *SelfModel) Save(path string) error {
	data := selfSaveData{
		KnownTopics:   s.KnownTopics,
		SuccessCount:  s.SuccessCount,
		FailureCount:  s.FailureCount,
		TotalInputs:   s.TotalInputs,
		AvgConfidence: s.AvgConfidence,
	}
	raw, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("self save marshal: %w", err)
	}
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, raw, 0600); err != nil {
		return fmt.Errorf("self save write: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("self save rename: %w", err)
	}
	return nil
}

// LoadSelfModel restores SelfModel state from a JSON file.
// Returns (nil, nil) if the file does not exist, allowing the caller
// to fall back to NewSelfModel.
func LoadSelfModel(path string) (*SelfModel, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("self load read: %w", err)
	}
	var data selfSaveData
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, fmt.Errorf("self load unmarshal: %w", err)
	}
	if data.KnownTopics == nil {
		data.KnownTopics = make(map[string]uint8)
	}
	return &SelfModel{
		KnownTopics:   data.KnownTopics,
		SuccessCount:  data.SuccessCount,
		FailureCount:  data.FailureCount,
		TotalInputs:   data.TotalInputs,
		AvgConfidence: data.AvgConfidence,
	}, nil
}
