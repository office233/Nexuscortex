package cortex

import (
	"encoding/json"
	"fmt"
	"os"
)

// ─────────────────────────────────────────────────────────────────────
// reward.go — Digital Dopamine System
// ─────────────────────────────────────────────────────────────────────
//
// The reward system provides the organism's motivation. It does NOT
// hardcode what is good or bad — it learns from prediction error.
//
// Positive reward = something good happened (prediction was correct,
//                   learned something new, novelty in the sweet spot)
// Negative reward = something bad happened (error, confusion, too
//                   much unpredictability)
//
// The Surprise() method creates the curiosity sweet spot: medium
// prediction error produces the largest positive reward (dopamine),
// while very high error produces pain (aversion).

// RewardSignal represents a single reward event delivered to the organism.
type RewardSignal struct {
	Value     int8   // -128 to +127: negative=pain, 0=neutral, positive=pleasure
	Source    string // What caused this signal (for debugging)
	Timestamp uint64 // When this happened
}

// RewardSystem tracks reward signals over time and derives motivation.
type RewardSystem struct {
	CurrentReward  int8   // Current reward level
	RewardHistory  []int8 // Last N reward values
	BaselineReward int8   // Running average (what's "normal")
	MaxHistory     int    // Maximum history length
	Config         Config // Runtime configuration
}

// NewRewardSystem creates a reward system with the given config.
func NewRewardSystem(cfg Config) *RewardSystem {
	maxHistory := cfg.RewardSystemCapacity
	if maxHistory < 1 {
		maxHistory = 1
	}
	return &RewardSystem{
		CurrentReward:  0,
		RewardHistory:  make([]int8, 0, maxHistory),
		BaselineReward: 0,
		MaxHistory:     maxHistory,
		Config:         cfg,
	}
}

// Signal delivers a reward signal to the system.
//
// The value is recorded in history, the current reward is updated, and
// the baseline is recalculated as a running average. Positive values
// mean something good happened (correct prediction, useful learning).
// Negative values mean something bad happened (error, confusion).
func (r *RewardSystem) Signal(value int8, source string) {
	r.CurrentReward = value

	// Append to history, evicting oldest if at capacity.
	if len(r.RewardHistory) >= r.MaxHistory {
		// Shift left by one position.
		copy(r.RewardHistory, r.RewardHistory[1:])
		r.RewardHistory[len(r.RewardHistory)-1] = value
	} else {
		r.RewardHistory = append(r.RewardHistory, value)
	}

	// Recalculate baseline as running average.
	r.updateBaseline()
}

// Surprise converts a prediction error magnitude into a reward value.
//
// This creates the curiosity sweet spot:
//
//   - Low error (< 30):    small positive (+10).
//     Confirming expectations is mildly good — stability.
//   - Medium error (30–100): larger positive (+40).
//     Novelty is interesting — dopamine! The organism seeks this.
//   - High error (> 100):   negative (−30).
//     Too much confusion — pain. The organism avoids this.
func (r *RewardSystem) Surprise(predictionError uint8) int8 {
	var reward int8

	ssLow := r.Config.RewardSweetSpotLow
	if ssLow == 0 {
		ssLow = 30
	}
	ssHigh := r.Config.RewardSweetSpotHigh
	if ssHigh == 0 {
		ssHigh = 100
	}
	lowVal := r.Config.RewardLowValue
	if lowVal == 0 {
		lowVal = 10
	}

	switch {
	case predictionError < ssLow:
		// Low error — confirming expectations is mildly rewarding.
		reward = lowVal
	case predictionError <= ssHigh:
		// Medium error — novelty sweet spot = dopamine!
		range_ := int16(ssHigh) - int16(ssLow)
		if range_ <= 0 {
			range_ = 1
		}
		t := int16(predictionError-ssLow) * 40 / range_
		reward = int8(20 + t)
	default:
		// High error — too confusing = pain.
		range_ := int16(255) - int16(ssHigh)
		if range_ <= 0 {
			range_ = 1
		}
		t := int16(predictionError-ssHigh) * 98 / range_
		reward = -int8(30 + t)
	}

	// Deliver the reward through the normal signal path.
	r.Signal(reward, "prediction_error")
	return reward
}

// GetDrive returns the current motivation level.
//
// Positive drive means the organism wants more of whatever is
// happening. Negative drive means it wants to avoid the current
// situation. Drive is the difference between current reward and
// baseline: if things are better than average, drive is positive.
func (r *RewardSystem) GetDrive() int8 {
	// Drive = current minus baseline, clamped to int8 range.
	drive := int16(r.CurrentReward) - int16(r.BaselineReward)
	if drive > 127 {
		drive = 127
	}
	if drive < -128 {
		drive = -128
	}
	return int8(drive)
}

// IsInFlow returns true if recent rewards are consistently positive.
//
// The organism is in a good learning state when it has enough history
// and the majority of recent rewards are positive. This corresponds
// to the psychological concept of "flow" — optimal challenge level.
func (r *RewardSystem) IsInFlow() bool {
	n := len(r.RewardHistory)
	if n < 3 {
		return false
	}

	// Check the most recent entries (up to last 8).
	window := n
	if window > 8 {
		window = 8
	}

	positiveCount := 0
	for i := n - window; i < n; i++ {
		if r.RewardHistory[i] > 0 {
			positiveCount++
		}
	}

	// Flow requires at least 75% positive rewards in the window.
	return positiveCount*4 >= window*3
}

// Reset clears all reward state back to neutral.
func (r *RewardSystem) Reset() {
	r.CurrentReward = 0
	r.RewardHistory = r.RewardHistory[:0]
	r.BaselineReward = 0
}

// updateBaseline recalculates the running-average baseline from history.
func (r *RewardSystem) updateBaseline() {
	n := len(r.RewardHistory)
	if n == 0 {
		r.BaselineReward = 0
		return
	}

	var sum int32
	for _, v := range r.RewardHistory {
		sum += int32(v)
	}
	r.BaselineReward = int8(sum / int32(n))
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save / Load
// ─────────────────────────────────────────────────────────────────────

// rewardSaveData is the JSON-serializable snapshot of RewardSystem state.
type rewardSaveData struct {
	CurrentReward  int8   `json:"current_reward"`
	RewardHistory  []int8 `json:"reward_history"`
	BaselineReward int8   `json:"baseline_reward"`
	MaxHistory     int    `json:"max_history"`
}

// Save persists the RewardSystem state to a JSON file.
func (r *RewardSystem) Save(path string) error {
	data := rewardSaveData{
		CurrentReward:  r.CurrentReward,
		RewardHistory:  r.RewardHistory,
		BaselineReward: r.BaselineReward,
		MaxHistory:     r.MaxHistory,
	}
	raw, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("reward save marshal: %w", err)
	}
	if err := os.WriteFile(path, raw, 0644); err != nil {
		return fmt.Errorf("reward save write: %w", err)
	}
	return nil
}

// LoadRewardSystem restores RewardSystem state from a JSON file.
// Returns (nil, nil) if the file does not exist, allowing the caller
// to fall back to NewRewardSystem.
func LoadRewardSystem(path string, cfg Config) (*RewardSystem, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reward load read: %w", err)
	}
	var data rewardSaveData
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, fmt.Errorf("reward load unmarshal: %w", err)
	}
	if data.RewardHistory == nil {
		data.RewardHistory = make([]int8, 0)
	}
	maxHistory := cfg.RewardSystemCapacity
	if maxHistory < 1 {
		maxHistory = 1
	}
	if len(data.RewardHistory) > maxHistory {
		data.RewardHistory = data.RewardHistory[len(data.RewardHistory)-maxHistory:]
	}
	return &RewardSystem{
		CurrentReward:  data.CurrentReward,
		RewardHistory:  data.RewardHistory,
		BaselineReward: data.BaselineReward,
		MaxHistory:     maxHistory,
		Config:         cfg,
	}, nil
}
