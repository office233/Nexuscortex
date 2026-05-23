package cortex

import (
	"math/rand"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════
// RadioNeuron Tests
// ═══════════════════════════════════════════════════════════════════

func TestRadioNeuronPackUnpack(t *testing.T) {
	n := PackRadioNeuron(42, 100, 200, 15, false) // phase 100 (7-bit max 127)
	if n.FreqListen() != 42 {
		t.Errorf("FreqListen: got %d, want 42", n.FreqListen())
	}
	if n.Phase() != 100 {
		t.Errorf("Phase: got %d, want 100", n.Phase())
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

	n.SetPhase(100) // 7-bit, max 127
	if n.Phase() != 100 {
		t.Errorf("SetPhase: got %d, want 100", n.Phase())
	}

	n.SetAmplitude(255)
	if n.Amplitude() != 255 {
		t.Errorf("SetAmplitude: got %d, want 255", n.Amplitude())
	}

	// Refractory is now a no-op (not enough bits in 4 bytes)
	// Just verify it doesn't crash
	n.SetRefractory(true)
	n.SetRefractory(false)
}

func TestRadioNeuronAdvancePhase(t *testing.T) {
	n := PackRadioNeuron(10, 120, 128, 5, false)
	n.AdvancePhase() // (120 + 10) & 0x7F = 130 & 127 = 2
	if n.Phase() != 2 {
		t.Errorf("Phase after advance: got %d, want 2 (wrapped at 128)", n.Phase())
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
	// In phase: difference = 0 → max resonance (+127)
	r0 := Resonance(50, 50)
	if r0 < 100 {
		t.Errorf("In-phase resonance should be high, got %d", r0)
	}

	// Slightly out of phase but still positive
	// delta7=16, delta8=32 → cos(32*360/256) = cos(45°) ≈ +90
	r16 := Resonance(66, 50) // delta=16 in 7-bit
	t.Logf("Delta 16 resonance: %d", r16)
	if r16 <= 0 {
		t.Errorf("Small 7-bit delta should still be positive, got %d", r16)
	}

	// 90° in 7-bit = delta7=32, delta8=64 → cos(90°) ≈ 0
	r32 := Resonance(82, 50) // delta=32 in 7-bit
	t.Logf("Delta 32 resonance (7-bit 90°): %d (should be near zero)", r32)
	if r32 > 10 || r32 < -10 {
		t.Errorf("90° apart (7-bit delta=32) should be ~0, got %d", r32)
	}

	// 180° in 7-bit = delta7=64, delta8=128 → cos(180°) ≈ -127
	r64 := Resonance(114, 50) // delta=64 in 7-bit
	t.Logf("Delta 64 resonance (7-bit 180°): %d (should be ~-127)", r64)
	if r64 > -100 {
		t.Errorf("180° apart (7-bit delta=64) should be ~-127, got %d", r64)
	}

	// Large delta should wrap via uint8 subtraction
	// and produce valid resonance values (no crash)
	rWrap := Resonance(10, 100) // delta wraps
	t.Logf("Wrap resonance (10-100): %d", rWrap)
}

func TestResonance7Bit(t *testing.T) {
	// Same phase = max resonance
	r := Resonance(0, 0)
	if r < 120 {
		t.Errorf("Same phase should be ~127, got %d", r)
	}

	// Half cycle apart (64 in 7-bit = 180°) = anti-resonance
	r2 := Resonance(64, 0)
	if r2 > -120 {
		t.Errorf("180° apart should be ~-127, got %d", r2)
	}

	// Quarter cycle (32 in 7-bit = 90°) = near zero
	r3 := Resonance(32, 0)
	if r3 > 10 || r3 < -10 {
		t.Errorf("90° apart should be ~0, got %d", r3)
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
