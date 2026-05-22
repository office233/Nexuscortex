package cortex

import (
	"encoding/json"
	"fmt"
	"os"
)

// ─────────────────────────────────────────────────────────────────────
// curiosity.go — Exploration Drive
// ─────────────────────────────────────────────────────────────────────
//
// Curiosity drives the organism to seek new information. It is NOT
// hardcoded — it emerges entirely from prediction-error statistics.
//
// The core insight is the Goldilocks principle of curiosity:
//   - Very low prediction errors → the organism is bored (nothing new).
//   - Medium prediction errors → the sweet spot (learnable novelty).
//   - Very high prediction errors → the organism is overwhelmed.
//
// The InterestMap tracks per-topic interest levels. Topics that
// consistently produce medium-range errors accumulate interest;
// topics that produce only low or only high errors lose interest.
//
// The ExplorationRate is a global scalar derived from the distribution
// of recent prediction errors across all topics. A diverse spread of
// errors signals a rich environment worth exploring; uniformly low or
// uniformly high errors signal stagnation or chaos, respectively.

// CuriosityDrive models an intrinsic motivation signal derived from
// prediction-error statistics, with no hardcoded curiosity targets.
type CuriosityDrive struct {
	ExplorationRate   uint8            // 0=exploit only, 255=explore everything
	InterestMap       map[string]uint8 // topic -> interest level
	BoredThreshold    uint8            // Below this error level = bored
	OverwhelmThresh   uint8            // Above this error level = overwhelmed
	InterestIncrement uint8            // How much interest grows per learnable observation
	InterestDecay     uint8            // How much interest shrinks per non-learnable observation
	RateStep          uint8            // How fast ExplorationRate changes
	RecentErrors      []uint8          // Recent prediction errors
	MaxErrorHistory   int
}

// NewCuriosityDrive creates a curiosity drive with the given config.
// Thresholds default to biologically plausible values: bored below 40, overwhelmed above 200.
func NewCuriosityDrive(cfg Config) *CuriosityDrive {
	maxHistory := cfg.CuriosityHistoryCapacity
	if maxHistory < 1 {
		maxHistory = 1
	}
	boredT := cfg.CuriosityBoredThreshold
	if boredT == 0 {
		boredT = 40
	}
	overwhelmT := cfg.CuriosityOverwhelmThreshold
	if overwhelmT == 0 {
		overwhelmT = 200
	}
	interestInc := cfg.CuriosityInterestIncrement
	if interestInc == 0 {
		interestInc = 10
	}
	interestDec := cfg.CuriosityInterestDecay
	if interestDec == 0 {
		interestDec = 5
	}
	rateStep := cfg.CuriosityRateStep
	if rateStep == 0 {
		rateStep = 10
	}
	return &CuriosityDrive{
		ExplorationRate: 128, // Start neutral.
		InterestMap:     make(map[string]uint8),
		BoredThreshold:  boredT,
		OverwhelmThresh: overwhelmT,
		InterestIncrement: interestInc,
		InterestDecay:   interestDec,
		RateStep:        rateStep,
		RecentErrors:    make([]uint8, 0, maxHistory),
		MaxErrorHistory: maxHistory,
	}
}

// ─────────────────────────────────────────────────────────────────────
// ObserveError — Core curiosity update
// ─────────────────────────────────────────────────────────────────────

// ObserveError processes a new prediction error on a topic and updates
// the curiosity state:
//
//  1. The error is appended to the rolling history.
//  2. Per-topic interest is adjusted:
//     - Errors in the "learnable zone" (between bored and overwhelm
//       thresholds) increase interest by 10 (capped at 255).
//     - Errors outside the zone decay interest by 5 (floored at 0).
//  3. ExplorationRate is recomputed from the variance of recent errors:
//     - All low errors  → ExplorationRate decreases (nothing new).
//     - All high errors → ExplorationRate decreases (overwhelmed).
//     - Mixed errors    → ExplorationRate increases (lots to learn).
func (c *CuriosityDrive) ObserveError(predictionError uint8, topic string) {
	// 1. Record the error in rolling history.
	if len(c.RecentErrors) >= c.MaxErrorHistory && c.MaxErrorHistory > 0 {
		// Drop the oldest entry (FIFO).
		c.RecentErrors = c.RecentErrors[1:]
	}
	c.RecentErrors = append(c.RecentErrors, predictionError)

	// 2. Update per-topic interest.
	interest := c.InterestMap[topic]
	if predictionError > c.BoredThreshold && predictionError < c.OverwhelmThresh {
		// Learnable zone: increase interest.
		if interest <= 255-c.InterestIncrement {
			interest += c.InterestIncrement
		} else {
			interest = 255
		}
	} else {
		// Outside zone: decay interest.
		if interest >= c.InterestDecay {
			interest -= c.InterestDecay
		} else {
			interest = 0
		}
	}
	c.InterestMap[topic] = interest

	// 3. Recompute ExplorationRate from recent error distribution.
	c.updateExplorationRate()
}

// updateExplorationRate sets ExplorationRate based on the distribution
// of recent prediction errors:
//
//   - Compute the mean of recent errors.
//   - Compute the mean absolute deviation (MAD) from that mean.
//   - If the mean is very low (< BoredThreshold) → low rate.
//   - If the mean is very high (> OverwhelmThresh) → low rate.
//   - Otherwise scale the rate by the deviation: high spread of errors
//     in the learnable zone → high exploration rate.
func (c *CuriosityDrive) updateExplorationRate() {
	n := len(c.RecentErrors)
	if n == 0 {
		return
	}

	// Mean of recent errors.
	var sum uint64
	for _, e := range c.RecentErrors {
		sum += uint64(e)
	}
	mean := uint8(sum / uint64(n))

	// Mean absolute deviation.
	var devSum uint64
	for _, e := range c.RecentErrors {
		if e > mean {
			devSum += uint64(e - mean)
		} else {
			devSum += uint64(mean - e)
		}
	}
	mad := uint8(devSum / uint64(n))

	// Determine exploration rate.
	if mean < c.BoredThreshold {
		// All low errors → nothing new to explore.
		if c.ExplorationRate >= c.RateStep {
			c.ExplorationRate -= c.RateStep
		} else {
			c.ExplorationRate = 0
		}
	} else if mean > c.OverwhelmThresh {
		// All high errors → too overwhelmed.
		if c.ExplorationRate >= c.RateStep {
			c.ExplorationRate -= c.RateStep
		} else {
			c.ExplorationRate = 0
		}
	} else {
		// Mixed errors in the learnable zone.
		// High deviation = more variety = more to explore.
		// Scale MAD into a rate adjustment (MAD max is ~127).
		boost := uint16(mad) * 2
		if boost > 255 {
			boost = 255
		}
		// EMA toward the boost value.
		c.ExplorationRate = uint8((uint16(c.ExplorationRate) + boost) / 2)
	}
}

// ─────────────────────────────────────────────────────────────────────
// Queries
// ─────────────────────────────────────────────────────────────────────

// ShouldExplore returns true when the exploration rate exceeds the
// midpoint, indicating the organism should seek novel inputs.
func (c *CuriosityDrive) ShouldExplore() bool {
	return c.ExplorationRate > 128
}

// MostInteresting returns the topic with the highest interest level,
// along with that level. Returns ("", 0) if no topics are tracked.
func (c *CuriosityDrive) MostInteresting() (string, uint8) {
	var bestTopic string
	var bestLevel uint8
	for topic, level := range c.InterestMap {
		if level > bestLevel {
			bestTopic = topic
			bestLevel = level
		}
	}
	return bestTopic, bestLevel
}

// MostBoring returns the topic with the lowest interest level,
// along with that level. Returns ("", 0) if no topics are tracked.
func (c *CuriosityDrive) MostBoring() (string, uint8) {
	var worstTopic string
	worstLevel := uint8(255)
	found := false
	for topic, level := range c.InterestMap {
		if !found || level < worstLevel {
			worstTopic = topic
			worstLevel = level
			found = true
		}
	}
	if !found {
		return "", 0
	}
	return worstTopic, worstLevel
}

// GetExplorationRate returns the current exploration rate (0–255).
func (c *CuriosityDrive) GetExplorationRate() uint8 {
	return c.ExplorationRate
}

// ─────────────────────────────────────────────────────────────────────
// Decay — Forgetting
// ─────────────────────────────────────────────────────────────────────

// Decay slowly reduces all interest levels, modeling the natural
// forgetting of topics that haven't been encountered recently.
// Each topic loses 1 point of interest per decay cycle. Topics that
// reach zero are removed from the interest map entirely.
func (c *CuriosityDrive) Decay() {
	for topic, level := range c.InterestMap {
		if level <= 1 {
			delete(c.InterestMap, topic)
		} else {
			c.InterestMap[topic] = level - 1
		}
	}

	// Also gently decay exploration rate toward neutral.
	if c.ExplorationRate > 128 {
		c.ExplorationRate--
	} else if c.ExplorationRate < 128 {
		c.ExplorationRate++
	}
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save / Load
// ─────────────────────────────────────────────────────────────────────

// curiositySaveData is the JSON-serializable snapshot of CuriosityDrive state.
type curiositySaveData struct {
	ExplorationRate   uint8            `json:"exploration_rate"`
	InterestMap       map[string]uint8 `json:"interest_map"`
	BoredThreshold    uint8            `json:"bored_threshold"`
	OverwhelmThresh   uint8            `json:"overwhelm_thresh"`
	InterestIncrement uint8            `json:"interest_increment,omitempty"`
	InterestDecay     uint8            `json:"interest_decay,omitempty"`
	RateStep          uint8            `json:"rate_step,omitempty"`
	RecentErrors      []uint8          `json:"recent_errors"`
	MaxErrorHistory   int              `json:"max_error_history"`
}

// Save persists the CuriosityDrive state to a JSON file.
func (c *CuriosityDrive) Save(path string) error {
	data := curiositySaveData{
		ExplorationRate:   c.ExplorationRate,
		InterestMap:       c.InterestMap,
		BoredThreshold:    c.BoredThreshold,
		OverwhelmThresh:   c.OverwhelmThresh,
		InterestIncrement: c.InterestIncrement,
		InterestDecay:     c.InterestDecay,
		RateStep:          c.RateStep,
		RecentErrors:      c.RecentErrors,
		MaxErrorHistory:   c.MaxErrorHistory,
	}
	raw, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("curiosity save marshal: %w", err)
	}
	if err := os.WriteFile(path, raw, 0644); err != nil {
		return fmt.Errorf("curiosity save write: %w", err)
	}
	return nil
}

// LoadCuriosityDrive restores CuriosityDrive state from a JSON file.
// Returns (nil, nil) if the file does not exist, allowing the caller
// to fall back to NewCuriosityDrive.
func LoadCuriosityDrive(path string, cfg Config) (*CuriosityDrive, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("curiosity load read: %w", err)
	}
	var data curiositySaveData
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, fmt.Errorf("curiosity load unmarshal: %w", err)
	}
	if data.InterestMap == nil {
		data.InterestMap = make(map[string]uint8)
	}
	if data.RecentErrors == nil {
		data.RecentErrors = make([]uint8, 0)
	}
	maxHistory := cfg.CuriosityHistoryCapacity
	if maxHistory < 1 {
		maxHistory = 1
	}
	if len(data.RecentErrors) > maxHistory {
		data.RecentErrors = data.RecentErrors[len(data.RecentErrors)-maxHistory:]
	}
	interestInc := data.InterestIncrement
	if interestInc == 0 {
		interestInc = 10
	}
	interestDec := data.InterestDecay
	if interestDec == 0 {
		interestDec = 5
	}
	rateStep := data.RateStep
	if rateStep == 0 {
		rateStep = 10
	}
	return &CuriosityDrive{
		ExplorationRate:   data.ExplorationRate,
		InterestMap:       data.InterestMap,
		BoredThreshold:    data.BoredThreshold,
		OverwhelmThresh:   data.OverwhelmThresh,
		InterestIncrement: interestInc,
		InterestDecay:     interestDec,
		RateStep:          rateStep,
		RecentErrors:      data.RecentErrors,
		MaxErrorHistory:   maxHistory,
	}, nil
}
