package cortex

import (
	"testing"
	"nexus-cortex/cortex/compute"
)

func TestFractalCortexNeurogenesis(t *testing.T) {
	cfg := DefaultConfig()
	engine := compute.NewCPUEngine()
	fc := NewFractalCortex(cfg, engine)

	if len(fc.Blocks) != 1 {
		t.Fatalf("expected 1 block initially, got %d", len(fc.Blocks))
	}

	// Trigger high error to spawn new block
	fc.CheckPredictionError(0.9)

	if len(fc.Blocks) != 2 {
		t.Fatalf("expected 2 blocks after neurogenesis, got %d", len(fc.Blocks))
	}

	// Lock should prevent immediate spawn
	fc.CheckPredictionError(0.9)
	if len(fc.Blocks) != 2 {
		t.Fatalf("expected lock to prevent spawn, got %d blocks", len(fc.Blocks))
	}

	// Test ProcessToken routing
	sdr := NewSDR(cfg.SDRSize)
	// Active a few bits
	for i := 0; i < 200; i++ {
		sdr.Bits[i/64] |= 1 << uint(i%64)
	}
	sdr.ActiveCount = 200

	outSDR := fc.ProcessToken(sdr)
	if outSDR.ActiveCount == 0 {
		t.Errorf("expected active bits from ProcessToken, got 0")
	}
}
