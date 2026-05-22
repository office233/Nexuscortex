package cortex

import (
	"math/rand"
	"os"
	"strings"
	"testing"
)

func TestOrganismNoEchoRegression(t *testing.T) {
	tempDir := "temp_org_test"
	t.Cleanup(func() {
		os.RemoveAll(tempDir)
	})

	// Initialize a fresh organism in a temporary directory with a Config
	rng := rand.New(rand.NewSource(42))
	cfg := DefaultConfig()
	cfg.DataDir = tempDir
	org := NewOrganism(cfg, rng)

	input := "where do neurons fire?"

	// Process the input when the organism is completely fresh (zero knowledge)
	response := org.Process(input)

	// Regression test: the input must not be echoed back.
	if strings.TrimSpace(response) == strings.TrimSpace(input) {
		t.Errorf("FAIL: input prompt %q was echoed back as the response!", input)
	}

	// Since there is no learned data, the response should fall back to the structured low-confidence policy:
	// "?"
	expected := "?"
	if response != expected {
		t.Errorf("expected low-confidence fallback %q for empty brain, got: %q", expected, response)
	}
}

func TestOrganismRecovery(t *testing.T) {
	tempDir := t.TempDir()

	rng := rand.New(rand.NewSource(42))
	cfg := DefaultConfig()
	cfg.DataDir = tempDir
	cfg.PrefrontalInputCurrent = 128
	cfg.HippoReconsolidationThresh = 200
	cfg.HippoInitialStrength = 10
	cfg.HippoLtpThreshold = 128

	org := NewOrganism(cfg, rng)

	// Teach the organism some facts
	org.Learn("câinele aleargă prin grădina mare")
	org.Process("unde aleargă câinele?")

	// Save organism to disk
	if err := org.Save(cfg.DataDir); err != nil {
		t.Fatalf("failed to save organism: %v", err)
	}

	// Load organism back
	loadedOrg, err := LoadOrganism(cfg, rng)
	if err != nil {
		t.Fatalf("failed to load organism: %v", err)
	}

	// Verify that configuration parameters are fully recovered
	if loadedOrg.Prefrontal.InputCurrent != cfg.PrefrontalInputCurrent {
		t.Errorf("expected Prefrontal.InputCurrent to be %d, got %d", cfg.PrefrontalInputCurrent, loadedOrg.Prefrontal.InputCurrent)
	}
	if loadedOrg.Hippocampus.ReconsolidationThresh != cfg.HippoReconsolidationThresh {
		t.Errorf("expected Hippocampus.ReconsolidationThresh to be %d, got %d", cfg.HippoReconsolidationThresh, loadedOrg.Hippocampus.ReconsolidationThresh)
	}
	if loadedOrg.Hippocampus.InitialStrength != cfg.HippoInitialStrength {
		t.Errorf("expected Hippocampus.InitialStrength to be %d, got %d", cfg.HippoInitialStrength, loadedOrg.Hippocampus.InitialStrength)
	}
	if loadedOrg.Hippocampus.LtpThreshold != cfg.HippoLtpThreshold {
		t.Errorf("expected Hippocampus.LtpThreshold to be %d, got %d", cfg.HippoLtpThreshold, loadedOrg.Hippocampus.LtpThreshold)
	}

	// Verify Prefrontal thinking is active (spiking with non-zero input current)
	testSDR := NewSDR(100)
	testSDR.Set(10)
	testSDR.Set(20)
	refined := loadedOrg.Prefrontal.Think(testSDR, 20)

	// Spiking reservoir with input current must generate activity
	activeCount := 0
	for i := 0; i < refined.Size; i++ {
		if refined.IsActive(i) {
			activeCount++
		}
	}
	if activeCount == 0 {
		t.Error("Prefrontal reservoir is completely silent (0 active bits) on loaded organism!")
	}
	// NOTE: Prefrontal confidence may be 0 with sparse spiking networks
	// that haven't developed stable attractors. This is correct behavior
	// after fixing measureStability to use Jaccard on active neurons only.
	// The old test relied on the buggy behavior where confidence was always
	// ~95% because it counted silent-silent neuron matches.

	// Verify Hippocampal episodic memory stores new distinct facts without collapsing existing ones
	initialCount := loadedOrg.Hippocampus.Size()
	if initialCount == 0 {
		t.Fatal("expected loaded Hippocampus to have stored memories, got 0")
	}

	// Storing a highly distinct new memory
	distinctInput := NewSDR(cfg.SDRSize)
	distinctInput.Set(1)
	distinctInput.Set(2)
	distinctOutput := NewSDR(cfg.SDRSize)
	distinctOutput.Set(3)
	distinctOutput.Set(4)

	loadedOrg.Hippocampus.Store(distinctInput, distinctOutput, "new facts")

	// Ensure the memory count increased (proving no LTP/Reconsolidation Collapse)
	newCount := loadedOrg.Hippocampus.Size()
	if newCount <= initialCount {
		t.Errorf("Hippocampus memory count did not increase (LTP/Reconsolidation collapse). Stored memories collapsed! Initial: %d, New: %d", initialCount, newCount)
	}
}

func TestOrganismLearnQADoesNotPersistDirectAnswerMap(t *testing.T) {
	tempDir := t.TempDir()
	rng := rand.New(rand.NewSource(42))
	cfg := DefaultConfig()
	cfg.DataDir = tempDir

	org := NewOrganism(cfg, rng)
	org.LearnQA("what stores episodic memories?", "the hippocampus stores episodic memories.")

	if err := org.Save(cfg.DataDir); err != nil {
		t.Fatalf("failed to save organism: %v", err)
	}

	raw, err := os.ReadFile(tempDir + "/organism.json")
	if err != nil {
		t.Fatalf("failed to read organism metadata: %v", err)
	}
	if strings.Contains(string(raw), "qa_memory") {
		t.Fatalf("organism metadata must not persist direct qa_memory: %s", string(raw))
	}
}

func TestOrganismLearnQARetrievesEpisodicAnswerWithoutQAMemory(t *testing.T) {
	tempDir := t.TempDir()
	rng := rand.New(rand.NewSource(42))
	cfg := DefaultConfig()
	cfg.DataDir = tempDir
	cfg.MaxGenWords = 0 // Disable Broca generation to test pure fallback
	cfg.HippocampusRecallThresh = 50 // Lower threshold since test questions are short

	org := NewOrganism(cfg, rng)
	org.FractalCortex = nil // Disable AGI generation to test pure associative recall
	org.LearnQA("who loves gamma?", "gamma loves delta.")
	org.LearnQA("where is alpha?", "alpha is stored in beta.")

	response := org.Process("where is alpha?")
	if strings.TrimSpace(response) != "alpha is stored in beta." {
		t.Fatalf("expected exact episodic answer recall, got %q", response)
	}

	if err := org.Save(cfg.DataDir); err != nil {
		t.Fatalf("failed to save organism: %v", err)
	}
	raw, err := os.ReadFile(tempDir + "/organism.json")
	if err != nil {
		t.Fatalf("failed to read organism metadata: %v", err)
	}
	if strings.Contains(string(raw), "qa_memory") {
		t.Fatalf("episodic QA recall must not persist qa_memory: %s", string(raw))
	}
}

func TestOrganismLoadIgnoresLegacyQAMemory(t *testing.T) {
	tempDir := t.TempDir()
	rng := rand.New(rand.NewSource(42))
	cfg := DefaultConfig()
	cfg.DataDir = tempDir

	org := NewOrganism(cfg, rng)
	if err := org.Save(cfg.DataDir); err != nil {
		t.Fatalf("failed to save organism: %v", err)
	}

	legacyMeta := `{"interaction_count":0,"qa_memory":{"what stores episodic memories ?":"LEGACY HARD CODED ANSWER"}}`
	if err := os.WriteFile(tempDir+"/organism.json", []byte(legacyMeta), 0644); err != nil {
		t.Fatalf("failed to write legacy organism metadata: %v", err)
	}

	loadedOrg, err := LoadOrganism(cfg, rand.New(rand.NewSource(42)))
	if err != nil {
		t.Fatalf("failed to load organism: %v", err)
	}

	response := loadedOrg.Process("what stores episodic memories?")
	if strings.Contains(response, "LEGACY HARD CODED ANSWER") {
		t.Fatalf("legacy qa_memory was used as a direct answer: %q", response)
	}
}

func TestOrganismSuppressesLowConfidenceUnknownResponse(t *testing.T) {
	tempDir := t.TempDir()
	rng := rand.New(rand.NewSource(42))
	cfg := DefaultConfig()
	cfg.DataDir = tempDir

	org := NewOrganism(cfg, rng)
	org.FractalCortex = nil // Disable AGI generation to test pure associative recall
	org.Learn("the hippocampus stores episodic memories")

	response := org.Process("who invented the imaginary tensor loom?")
	if response != "(no confident response)" {
		t.Fatalf("expected unknown prompt to produce low-confidence fallback, got %q", response)
	}
}
