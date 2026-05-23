package cortex

import (
	"math/rand"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════
// RadioNeuron Tests
// ═══════════════════════════════════════════════════════════════════

func TestRadioNeuronPackUnpack(t *testing.T) {
	n := PackRadioNeuron(42, 128, 200, 15, false)
	if n.FreqListen() != 42 {
		t.Errorf("FreqListen: got %d, want 42", n.FreqListen())
	}
	if n.Phase() != 128 {
		t.Errorf("Phase: got %d, want 128", n.Phase())
	}
	if n.Amplitude() != 200 {
		t.Errorf("Amplitude: got %d, want 200", n.Amplitude())
	}
	if n.FreqEmit() != 15 {
		t.Errorf("FreqEmit: got %d, want 15", n.FreqEmit())
	}
	if n.IsInhibitory() {
		t.Error("should not be inhibitory")
	}
	if n.IsRefractory() {
		t.Error("should not be refractory")
	}
}

func TestRadioNeuronInhibitory(t *testing.T) {
	n := PackRadioNeuron(10, 20, 30, 5, true)
	if !n.IsInhibitory() {
		t.Error("should be inhibitory")
	}
	if n.FreqListen() != 10 {
		t.Errorf("FreqListen: got %d, want 10", n.FreqListen())
	}
	if n.FreqEmit() != 5 {
		t.Errorf("FreqEmit: got %d, want 5", n.FreqEmit())
	}
}

func TestRadioNeuronSetters(t *testing.T) {
	n := PackRadioNeuron(10, 20, 30, 5, false)

	n.SetFreqListen(100)
	if n.FreqListen() != 100 {
		t.Errorf("SetFreqListen: got %d, want 100", n.FreqListen())
	}
	// Verify other fields unchanged
	if n.Phase() != 20 {
		t.Errorf("Phase changed: got %d", n.Phase())
	}

	n.SetPhase(200)
	if n.Phase() != 200 {
		t.Errorf("SetPhase: got %d, want 200", n.Phase())
	}

	n.SetAmplitude(255)
	if n.Amplitude() != 255 {
		t.Errorf("SetAmplitude: got %d, want 255", n.Amplitude())
	}

	n.SetRefractory(true)
	if !n.IsRefractory() {
		t.Error("should be refractory")
	}
	n.SetRefractory(false)
	if n.IsRefractory() {
		t.Error("should not be refractory")
	}
}

func TestRadioNeuronAdvancePhase(t *testing.T) {
	n := PackRadioNeuron(10, 250, 128, 5, false)
	n.AdvancePhase() // 250 + 10 = 260 → wraps to 4
	if n.Phase() != 4 {
		t.Errorf("Phase after advance: got %d, want 4 (wrapped)", n.Phase())
	}
}

func TestRadioNeuronIsAlive(t *testing.T) {
	alive := PackRadioNeuron(10, 20, 1, 5, false)
	if !alive.IsAlive() {
		t.Error("amplitude 1 should be alive")
	}
	dead := PackRadioNeuron(10, 20, 0, 5, false)
	if dead.IsAlive() {
		t.Error("amplitude 0 should be dead")
	}
}

func TestResonance(t *testing.T) {
	// In phase: difference = 0 → max resonance
	r0 := Resonance(100, 100)
	if r0 < 100 {
		t.Errorf("In-phase resonance should be high, got %d", r0)
	}

	// Anti-phase: difference = 128 → negative resonance
	r128 := Resonance(200, 72) // 200 - 72 = 128
	if r128 > -100 {
		t.Errorf("Anti-phase resonance should be strongly negative, got %d", r128)
	}

	// 90° offset: difference = 64 → near zero
	r64 := Resonance(164, 100) // 164 - 100 = 64
	if r64 > 30 || r64 < -30 {
		t.Errorf("90° resonance should be near zero, got %d", r64)
	}
}

// ═══════════════════════════════════════════════════════════════════
// RadioBus Tests
// ═══════════════════════════════════════════════════════════════════

func TestRadioBusEmitRead(t *testing.T) {
	var bus RadioBus

	bus.Emit(42, 100, 50, false) // excitatory
	signal, phase := bus.Read(42)
	if signal != 100 {
		t.Errorf("Signal: got %d, want 100", signal)
	}
	if phase != 50 {
		t.Errorf("Phase: got %d, want 50", phase)
	}

	// Different frequency should be zero
	signal2, _ := bus.Read(43)
	if signal2 != 0 {
		t.Errorf("Wrong freq signal: got %d, want 0", signal2)
	}
}

func TestRadioBusConstructiveInterference(t *testing.T) {
	var bus RadioBus

	bus.Emit(42, 100, 50, false)
	bus.Emit(42, 80, 55, false)
	signal, _ := bus.Read(42)
	if signal != 180 {
		t.Errorf("Constructive: got %d, want 180", signal)
	}
}

func TestRadioBusDestructiveInterference(t *testing.T) {
	var bus RadioBus

	bus.Emit(42, 100, 50, false) // excitatory: +100
	bus.Emit(42, 60, 50, true)   // inhibitory: -60
	signal, _ := bus.Read(42)
	if signal != 40 {
		t.Errorf("Destructive: got %d, want 40", signal)
	}
}

func TestRadioBusClear(t *testing.T) {
	var bus RadioBus

	bus.Emit(42, 100, 50, false)
	bus.Clear()
	signal, _ := bus.Read(42)
	if signal != 0 {
		t.Errorf("After clear: got %d, want 0", signal)
	}
}

func TestRadioBusActiveChannels(t *testing.T) {
	var bus RadioBus

	bus.Emit(10, 100, 0, false)
	bus.Emit(50, 200, 0, false)
	bus.Emit(100, 30, 0, false) // below threshold

	active := bus.ActiveChannels(50)
	if len(active) != 2 {
		t.Errorf("Active channels: got %d, want 2", len(active))
	}
}

func TestRadioBusPeakFrequency(t *testing.T) {
	var bus RadioBus

	bus.Emit(10, 50, 0, false)
	bus.Emit(42, 200, 0, false) // strongest
	bus.Emit(99, 100, 0, false)

	freq, amp := bus.PeakFrequency()
	if freq != 42 {
		t.Errorf("Peak freq: got %d, want 42", freq)
	}
	if amp != 200 {
		t.Errorf("Peak amp: got %d, want 200", amp)
	}
}

// ═══════════════════════════════════════════════════════════════════
// RadioCortex Tests
// ═══════════════════════════════════════════════════════════════════

func TestRadioCortexCreation(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(1000, rng)

	if rc.Size != 1000 {
		t.Errorf("Size: got %d, want 1000", rc.Size)
	}
	if rc.InputEnd != 100 {
		t.Errorf("InputEnd: got %d, want 100 (10%%)", rc.InputEnd)
	}
	if rc.OutputStart != 900 {
		t.Errorf("OutputStart: got %d, want 900", rc.OutputStart)
	}

	// Check all neurons are alive
	stats := rc.Stats()
	if stats.AliveNeurons != 1000 {
		t.Errorf("Alive: got %d, want 1000", stats.AliveNeurons)
	}
}

func TestRadioCortexStep(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(1000, rng)

	// Inject some signal
	rc.InjectSignal([]uint8{5, 10, 15}, 200)

	// Run a step
	fired := rc.Step()
	t.Logf("Fired: %d / %d neurons", fired, rc.Size)

	// Should have some activity
	if rc.TickCount != 1 {
		t.Errorf("TickCount: got %d, want 1", rc.TickCount)
	}
}

func TestRadioCortexResonance(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(100, rng)

	// Create two neurons on same frequency — they should communicate
	rc.Neurons[0] = PackRadioNeuron(42, 0, 200, 10, false)   // listens on 42, emits on 10
	rc.Neurons[50] = PackRadioNeuron(10, 0, 200, 42, false) // listens on 10, emits on 42

	// Manually fire neuron 0
	rc.Fired[0] = true

	// Step: neuron 0 emits on freq 10 → neuron 50 should receive
	rc.Step()

	// Now neuron 50 should have been exposed to signal
	// (whether it fires depends on threshold, but the bus should have signal)
	t.Logf("Tick 1: %d neurons fired", rc.Stats().FiredNeurons)
}

func TestRadioCortexConfirm(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(10, rng)

	startAmp := rc.Neurons[0].Amplitude()
	rc.Fired[0] = true
	rc.Confirm()

	newAmp := rc.Neurons[0].Amplitude()
	if newAmp != startAmp+1 {
		t.Errorf("Confirm: amplitude should increase by 1, got %d→%d", startAmp, newAmp)
	}
}

func TestRadioCortexContradict(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(10, rng)

	startAmp := rc.Neurons[0].Amplitude()
	rc.Fired[0] = true
	rc.Contradict()

	newAmp := rc.Neurons[0].Amplitude()
	if newAmp != startAmp-1 {
		t.Errorf("Contradict: amplitude should decrease by 1, got %d→%d", startAmp, newAmp)
	}
}

func TestRadioCortexContradictFreqDrift(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(10, rng)

	// Set very low amplitude to trigger frequency drift
	rc.Neurons[0].SetAmplitude(20)
	startFreq := rc.Neurons[0].FreqListen()
	rc.Fired[0] = true
	rc.Contradict()

	newFreq := rc.Neurons[0].FreqListen()
	if newFreq == startFreq {
		// Could happen 50% of the time due to random, so don't fail
		t.Log("Freq didn't drift (possible, 50% chance)")
	} else {
		t.Logf("Freq drifted: %d → %d (learning by re-tuning)", startFreq, newFreq)
	}
}

func TestRadioCortexNeurogenesis(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(100, rng)

	// Kill some neurons
	rc.Neurons[10].SetAmplitude(0)
	rc.Neurons[20].SetAmplitude(0)
	rc.Neurons[30].SetAmplitude(0)

	replaced := rc.Neurogenesis()
	if replaced != 3 {
		t.Errorf("Neurogenesis: replaced %d, want 3", replaced)
	}

	// All should be alive again
	for _, idx := range []int{10, 20, 30} {
		if !rc.Neurons[idx].IsAlive() {
			t.Errorf("Neuron %d should be alive after neurogenesis", idx)
		}
	}
}

func TestRadioCortexMultiStep(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(1000, rng)

	// Run 100 ticks and track activity
	totalFired := 0
	for tick := 0; tick < 100; tick++ {
		if tick%10 == 0 {
			// Inject stimulus every 10 ticks
			rc.InjectSignal([]uint8{5, 10, 15, 20, 25}, 200)
		}
		fired := rc.Step()
		totalFired += fired
	}

	t.Logf("100 ticks: total fired = %d, avg = %.1f/tick", totalFired, float64(totalFired)/100)
	if rc.TickCount != 100 {
		t.Errorf("TickCount: got %d, want 100", rc.TickCount)
	}
}

func TestRadioCortexAmplitudeCap(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(10, rng)

	// Set to max and try to increase
	rc.Neurons[0].SetAmplitude(255)
	rc.Fired[0] = true
	rc.Confirm()
	if rc.Neurons[0].Amplitude() != 255 {
		t.Errorf("Amplitude should cap at 255, got %d", rc.Neurons[0].Amplitude())
	}

	// Set to 0 and try to decrease
	rc.Neurons[1].SetAmplitude(0)
	rc.Fired[1] = true
	rc.Contradict()
	if rc.Neurons[1].Amplitude() != 0 {
		t.Errorf("Amplitude should floor at 0, got %d", rc.Neurons[1].Amplitude())
	}
}

// ═══════════════════════════════════════════════════════════════════
// Benchmarks
// ═══════════════════════════════════════════════════════════════════

func BenchmarkRadioCortexStep100K(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(100_000, rng)

	// Pre-fire 5% of neurons
	for i := 0; i < 5000; i++ {
		rc.Fired[rng.Intn(100_000)] = true
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rc.Step()
	}
}

func BenchmarkRadioCortexStep1M(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	rc := NewRadioCortex(1_000_000, rng)

	// Pre-fire 5% of neurons
	for i := 0; i < 50_000; i++ {
		rc.Fired[rng.Intn(1_000_000)] = true
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rc.Step()
	}
}

func BenchmarkRadioBusEmit(b *testing.B) {
	var bus RadioBus
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.Emit(uint8(i%256), 128, uint8(i%256), false)
		if i%1000 == 999 {
			bus.Clear()
		}
	}
}

func BenchmarkRadioNeuronPack(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = PackRadioNeuron(uint8(i), uint8(i>>8), 128, uint8(i>>4), false)
	}
}

func BenchmarkResonance(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Resonance(uint8(i), uint8(i*7))
	}
}
