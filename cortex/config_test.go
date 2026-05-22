package cortex

import (
	"testing"
)

func TestConfigValidation(t *testing.T) {
	// 1. Default config should validate successfully
	cfg := DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Errorf("expected default config to be valid, got error: %v", err)
	}

	// 2. Empty WebBindAddr should be rejected
	cfg = DefaultConfig()
	cfg.WebBindAddr = ""
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for empty WebBindAddr, got nil")
	}

	// 3. WebLearnerTimeoutSecs > 120 should be rejected
	cfg = DefaultConfig()
	cfg.WebLearnerTimeoutSecs = 150
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for WebLearnerTimeoutSecs > 120, got nil")
	}

	// 4. WebLearnerBodyLimitMB > 100 should be rejected
	cfg = DefaultConfig()
	cfg.WebLearnerBodyLimitMB = 150
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for WebLearnerBodyLimitMB > 100, got nil")
	}

	// 5. Invalid ThousandBrainsColConnectivity should be rejected
	cfg = DefaultConfig()
	cfg.ThousandBrainsColConnectivity = 1.5
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for ThousandBrainsColConnectivity > 1.0, got nil")
	}

	// 6. Invalid capacities should be rejected
	cfg = DefaultConfig()
	cfg.RewardSystemCapacity = 0
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for RewardSystemCapacity <= 0, got nil")
	}

	cfg = DefaultConfig()
	cfg.EmotionHistoryCapacity = -5
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for EmotionHistoryCapacity <= 0, got nil")
	}

	// 7. Invalid WebGPUTimeoutSecs should be rejected
	cfg = DefaultConfig()
	cfg.WebGPUTimeoutSecs = 200
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for WebGPUTimeoutSecs > 120, got nil")
	}

	// 8. QuantumMultiSamples < 1 should be rejected
	cfg = DefaultConfig()
	cfg.QuantumMultiSamples = 0
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for QuantumMultiSamples < 1, got nil")
	}

	// 9. QuantumMultiSamples > 16 should be rejected
	cfg = DefaultConfig()
	cfg.QuantumMultiSamples = 17
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for QuantumMultiSamples > 16, got nil")
	}

	// 10. Default quantum config values should be valid
	cfg = DefaultConfig()
	cfg.EnableQuantumInspired = true
	cfg.QuantumTemperature = 128
	cfg.QuantumMultiSamples = 4
	if err := cfg.Validate(); err != nil {
		t.Errorf("expected quantum config to be valid, got error: %v", err)
	}
}
