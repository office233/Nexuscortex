package cortex

import (
	"math/rand"
	"testing"
)

// BenchmarkNeuronPoolStep benchmarks the StepAll method of a large NeuronPool
func BenchmarkNeuronPoolStep(b *testing.B) {
	pool := NewNeuronPool(3000)
	for i := 0; i < 3000; i++ {
		pool.AddNeuron(NewSpikingNeuron(NTypeExcitatory, RegionWorkspace, 120))
	}
	inputs := make([]uint8, 3000)
	for i := range inputs {
		inputs[i] = 150
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.StepAll(inputs)
	}
}

// BenchmarkNetworkTick benchmarks the Tick method of a full 3000-neuron reservoir SNN
func BenchmarkNetworkTick(b *testing.B) {
	cfg := DefaultConfig()
	rng := rand.New(rand.NewSource(42))
	// 3000 neurons with 5% sparse connectivity matching prefrontal defaults
	net := NewNetwork(3000, 0.05, cfg, rng)

	inputs := make([]uint8, 3000)
	for i := 0; i < 50; i++ {
		inputs[i] = 128
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.Tick(inputs)
	}
}

// BenchmarkPrefrontalThink benchmarks the Think method of Prefrontal reasoning module
func BenchmarkPrefrontalThink(b *testing.B) {
	cfg := DefaultConfig()
	rng := rand.New(rand.NewSource(42))
	prefrontal := NewPrefrontalWithSize(cfg, 3000, 0.05, rng)

	inputSDR := NewSDR(10000)
	for i := 0; i < 50; i++ {
		inputSDR.Set(i * 15)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prefrontal.Think(inputSDR, 10)
	}
}

// BenchmarkThousandBrainsProcess benchmarks the parallel Process method of ThousandBrains
func BenchmarkThousandBrainsProcess(b *testing.B) {
	cfg := DefaultConfig()
	rng := rand.New(rand.NewSource(42))
	tb := NewThousandBrains(cfg, rng)
	input := NewSDR(cfg.ThousandBrainsColNeurons)
	for i := 0; i < 10; i++ {
		input.Set(i * 7)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tb.Process(input)
	}
}
