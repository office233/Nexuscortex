package cortex

import (
	"math/rand"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────
// error_learning_test.go — Tests for Error-Driven Synaptic Learning
// ─────────────────────────────────────────────────────────────────────

func TestErrorDrivenLearnerBasic(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ErrorLearningStrength = 10
	cfg.ErrorLearningThreshold = 100

	// Use a small network for predictable testing.
	cfg.PrefrontalNetSize = 100
	cfg.SynapseMinWeight = 50
	cfg.SynapseMaxWeight = 50
	cfg.SynapseMinDelay = 1
	cfg.SynapseMaxDelay = 1
	cfg.NeuronMinThreshold = 100
	cfg.NeuronMaxThreshold = 200
	cfg.NeuronMinLeak = 1
	cfg.NeuronMaxLeak = 4

	rng := rand.New(rand.NewSource(42))
	net := NewNetwork(100, 0.1, cfg, rng)

	// Record initial total weight.
	var initialTotalWeight uint64
	for i := range net.Synapses {
		initialTotalWeight += uint64(net.Synapses[i].Weight)
	}

	// Create predicted and actual SDRs with some mismatch.
	predicted := NewSDR(100)
	actual := NewSDR(100)

	// Set overlapping bits (correct predictions).
	for i := 0; i < 10; i++ {
		predicted.Set(i)
		actual.Set(i)
	}

	// False positives: predicted=1 but actual=0 (bits 10-19).
	for i := 10; i < 20; i++ {
		predicted.Set(i)
	}

	// False negatives: predicted=0 but actual=1 (bits 20-29).
	for i := 20; i < 30; i++ {
		actual.Set(i)
	}

	learner := NewErrorDrivenLearner(cfg)
	// Use high prediction error (>200) to trigger full adjustment.
	learner.AdjustSynapses(net, predicted, actual, 220)

	// Verify synapses changed: total weight should differ.
	var newTotalWeight uint64
	for i := range net.Synapses {
		newTotalWeight += uint64(net.Synapses[i].Weight)
	}

	if newTotalWeight == initialTotalWeight {
		t.Errorf("Expected synapse weights to change with high prediction error, but total weight unchanged: %d", initialTotalWeight)
	}

	// Verify that false-positive target neurons had synapses weakened.
	weakened := false
	for i := range net.Synapses {
		tgt := int(net.Synapses[i].Target)
		if tgt >= 10 && tgt < 20 && net.Synapses[i].Weight < 50 {
			weakened = true
			break
		}
	}
	if !weakened {
		t.Error("Expected some synapses feeding false-positive neurons (10-19) to be weakened")
	}

	// Verify that false-negative target neurons had synapses strengthened.
	strengthened := false
	for i := range net.Synapses {
		tgt := int(net.Synapses[i].Target)
		if tgt >= 20 && tgt < 30 && net.Synapses[i].Weight > 50 {
			strengthened = true
			break
		}
	}
	if !strengthened {
		t.Error("Expected some synapses feeding false-negative neurons (20-29) to be strengthened")
	}
}

func TestErrorDrivenLearnerNoChangeOnLowError(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ErrorLearningStrength = 10
	cfg.ErrorLearningThreshold = 100

	cfg.PrefrontalNetSize = 50
	cfg.SynapseMinWeight = 50
	cfg.SynapseMaxWeight = 50
	cfg.SynapseMinDelay = 1
	cfg.SynapseMaxDelay = 1
	cfg.NeuronMinThreshold = 100
	cfg.NeuronMaxThreshold = 200
	cfg.NeuronMinLeak = 1
	cfg.NeuronMaxLeak = 4

	rng := rand.New(rand.NewSource(42))
	net := NewNetwork(50, 0.1, cfg, rng)

	// Snapshot weights before.
	weightsBefore := make([]uint8, len(net.Synapses))
	for i := range net.Synapses {
		weightsBefore[i] = net.Synapses[i].Weight
	}

	// Create mismatched SDRs.
	predicted := NewSDR(50)
	actual := NewSDR(50)
	for i := 0; i < 10; i++ {
		predicted.Set(i)
	}
	for i := 10; i < 20; i++ {
		actual.Set(i)
	}

	learner := NewErrorDrivenLearner(cfg)
	// Use low prediction error (< threshold) — should NOT adjust.
	learner.AdjustSynapses(net, predicted, actual, 50)

	// Verify no changes.
	for i := range net.Synapses {
		if net.Synapses[i].Weight != weightsBefore[i] {
			t.Errorf("Synapse %d weight changed from %d to %d with low error; expected no change",
				i, weightsBefore[i], net.Synapses[i].Weight)
			break
		}
	}
}

func TestErrorDrivenLearnerRespectsCaps(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ErrorLearningStrength = 50 // Aggressive for testing
	cfg.ErrorLearningThreshold = 100
	cfg.StdpMaxWeight = 200 // Cap below 255

	cfg.PrefrontalNetSize = 20
	cfg.SynapseMinWeight = 190 // Start near the cap
	cfg.SynapseMaxWeight = 190
	cfg.SynapseMinDelay = 1
	cfg.SynapseMaxDelay = 1
	cfg.NeuronMinThreshold = 100
	cfg.NeuronMaxThreshold = 200
	cfg.NeuronMinLeak = 1
	cfg.NeuronMaxLeak = 4

	rng := rand.New(rand.NewSource(42))
	net := NewNetwork(20, 0.2, cfg, rng)

	// Create false negatives on all neurons to trigger strengthening.
	predicted := NewSDR(20)
	actual := NewSDR(20)
	for i := 0; i < 20; i++ {
		actual.Set(i)
	}

	learner := NewErrorDrivenLearner(cfg)
	learner.AdjustSynapses(net, predicted, actual, 250)

	// Verify no weight exceeds StdpMaxWeight.
	for i := range net.Synapses {
		if net.Synapses[i].Weight > cfg.StdpMaxWeight {
			t.Errorf("Synapse %d weight %d exceeds StdpMaxWeight cap %d",
				i, net.Synapses[i].Weight, cfg.StdpMaxWeight)
		}
	}
}

func TestErrorDrivenLearnerMediumError(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ErrorLearningStrength = 10
	cfg.ErrorLearningThreshold = 100

	cfg.PrefrontalNetSize = 50
	cfg.SynapseMinWeight = 100
	cfg.SynapseMaxWeight = 100
	cfg.SynapseMinDelay = 1
	cfg.SynapseMaxDelay = 1
	cfg.NeuronMinThreshold = 100
	cfg.NeuronMaxThreshold = 200
	cfg.NeuronMinLeak = 1
	cfg.NeuronMaxLeak = 4

	rng := rand.New(rand.NewSource(42))
	net := NewNetwork(50, 0.1, cfg, rng)

	// Create false positives on neurons 0-9.
	predicted := NewSDR(50)
	actual := NewSDR(50)
	for i := 0; i < 10; i++ {
		predicted.Set(i)
	}

	learner := NewErrorDrivenLearner(cfg)
	// Medium error (150) → half strength = 5.
	learner.AdjustSynapses(net, predicted, actual, 150)

	// Verify adjustment was applied at half strength (5, not 10).
	for i := range net.Synapses {
		tgt := int(net.Synapses[i].Target)
		if tgt >= 0 && tgt < 10 {
			// Should be weakened by 5 (half of 10), so weight = 95.
			if net.Synapses[i].Weight == 95 {
				return // found at least one correctly adjusted synapse
			}
		}
	}
	t.Error("Expected at least one false-positive synapse weakened by half-strength (5), from 100 to 95")
}

func TestErrorDrivenLearnerSkipsEmptyNetwork(t *testing.T) {
	cfg := DefaultConfig()
	learner := NewErrorDrivenLearner(cfg)
	net := &Network{Neurons: NewNeuronPool(0)}

	predicted := NewSDR(16)
	actual := NewSDR(16)
	predicted.Set(1)
	actual.Set(2)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("AdjustSynapses panicked for an empty network: %v", r)
		}
	}()

	learner.AdjustSynapses(net, predicted, actual, 255)
}
