package cortex

import (
	"math/rand"
	"testing"
)

func TestHomeostaticThresholdPlasticity(t *testing.T) {
	// Create a neuron with threshold 120
	n := NewSpikingNeuron(NTypeExcitatory, RegionWorkspace, 120)

	// Step it with input current 200 so it spikes
	fired := n.Step(200, 15, 8, 80, 1)
	if !fired {
		t.Fatal("Neuron should spike under high current")
	}

	// The threshold G should increase on spike (+8)
	if n.Threshold() != 128 {
		t.Errorf("Expected threshold to increase to 128, got %d", n.Threshold())
	}

	// A spike sets the refractory counter to 15. We step it 15 times to clear refractory.
	for i := 0; i < 15; i++ {
		n.Step(0, 15, 8, 80, 1)
	}

	// Refractory counter should now be 0, and voltage stays at 0.
	// Step it once more with 0 input. It should not spike, so G should decay by 1.
	n.Step(0, 15, 8, 80, 1)
	if n.Threshold() != 127 {
		t.Errorf("Expected threshold to decay to 127 on silent step, got %d", n.Threshold())
	}

	// Step it many times to see if it decays down to floor 80
	for i := 0; i < 100; i++ {
		n.Step(0, 15, 8, 80, 1)
	}
	if n.Threshold() != 80 {
		t.Errorf("Expected threshold to decay to floor 80, got %d", n.Threshold())
	}
}

func TestWTALateralInhibition(t *testing.T) {
	cfg := DefaultConfig()
	rng := rand.New(rand.NewSource(42))
	// Create a network of size 200 (2 columns of 100 neurons)
	net := NewNetwork(200, 0.05, cfg, rng)

	// Inject input to Column 0 (neurons 0-99) and Column 1 (neurons 100-199)
	// We want Column 0 to be more active.
	// Inject pattern on tick 1 by running the network with inputs
	inputs := make([]uint8, 200)
	for i := 0; i < 50; i++ {
		inputs[i] = 150 // Column 0 active
	}

	net.Tick(inputs) // First tick: generates spikes in Column 0, records to SpikeHistory

	// Run second tick with empty input: this processes the previous spikes and updates ColSpikeTraces
	net.Tick(nil)

	// Traces should be initialized and Column 0 trace should be non-zero
	numCols := len(net.ColSpikeTraces)
	if numCols != 2 {
		t.Fatalf("Expected 2 columns, got %d", numCols)
	}

	col0Trace := net.ColSpikeTraces[0]
	if col0Trace == 0 {
		t.Error("Expected Column 0 activation trace to be updated after step propagation")
	}
}

func TestMetacognitiveReset(t *testing.T) {
	cfg := DefaultConfig()
	rng := rand.New(rand.NewSource(42))
	// Create a small network
	net := NewNetwork(100, 0.1, cfg, rng)

	// Direct injection of ChaosCounter to simulate prolonged chaotic state trigger
	net.ChaosCounter = 8

	// Inject massive current to all neurons to induce chaos
	inputs := make([]uint8, 100)
	for i := 0; i < 100; i++ {
		inputs[i] = 250
	}

	// This Tick will spike all neurons (100% spikes, which is >30%), causing ChaosCounter to become 9.
	// Since ChaosCounter > 8, it triggers the Noradrenergic Metacognitive Reset!
	net.Tick(inputs)

	// Verify MetacognitiveDampening is active (which means reset was triggered)
	if net.MetacognitiveDampening == 0 {
		t.Error("Expected MetacognitiveDampening to be active after prolonged chaos trigger")
	}

	// Check if thresholds got inflated
	for i := 0; i < 100; i++ {
		neuron := net.Neurons.GetNeuron(i)
		if neuron != nil && neuron.Threshold() < 110 {
			t.Errorf("Expected neuron %d threshold to be inflated, got %d", i, neuron.Threshold())
		}
	}
}
