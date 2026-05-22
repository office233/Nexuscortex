package cortex

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// TestPersistenceBrainOversizedCount verifies that LoadBrain rejects
// a file with an absurdly large synapse count (OOM protection).
func TestPersistenceBrainOversizedCount(t *testing.T) {
	dir := t.TempDir()
	brainPath := filepath.Join(dir, "bad.nxbrain")
	vocabPath := filepath.Join(dir, "vocab.json")

	// Write a minimal valid vocab file.
	if err := os.WriteFile(vocabPath, []byte(`{"word_to_id":{},"id_to_word":{},"next_id":1}`), 0644); err != nil {
		t.Fatal(err)
	}

	// Write a brain file with valid magic but absurd count.
	f, err := os.Create(brainPath)
	if err != nil {
		t.Fatal(err)
	}
	f.Write([]byte(BrainFileMagic))                             // 8 bytes magic
	binary.Write(f, binary.LittleEndian, uint32(4_000_000_000)) // 4 billion synapses
	f.Close()

	_, err = LoadBrain(brainPath, vocabPath, nil, DefaultConfig())
	if err == nil {
		t.Fatal("expected error for oversized synapse count, got nil")
	}
	t.Logf("correctly rejected: %v", err)
}

// TestPersistenceBrainValidFile verifies that a normal brain file still loads.
func TestPersistenceBrainValidFile(t *testing.T) {
	dir := t.TempDir()
	brainPath := filepath.Join(dir, "good.nxbrain")
	vocabPath := filepath.Join(dir, "vocab.json")

	if err := os.WriteFile(vocabPath, []byte(`{"word_to_id":{},"id_to_word":{},"next_id":1}`), 0644); err != nil {
		t.Fatal(err)
	}

	f, err := os.Create(brainPath)
	if err != nil {
		t.Fatal(err)
	}
	f.Write([]byte(BrainFileMagic))
	binary.Write(f, binary.LittleEndian, uint32(0)) // zero synapses
	f.Close()

	brain, err := LoadBrain(brainPath, vocabPath, nil, DefaultConfig())
	if err != nil {
		t.Fatalf("expected valid load, got error: %v", err)
	}
	if len(brain.Synapses) != 0 {
		t.Errorf("expected 0 synapses, got %d", len(brain.Synapses))
	}
}

// TestPersistenceHippocampusOversizedCount verifies LoadHippocampus
// rejects a file with an absurdly large memory count.
func TestPersistenceHippocampusOversizedCount(t *testing.T) {
	dir := t.TempDir()
	hipPath := filepath.Join(dir, "bad.nxhip")

	f, err := os.Create(hipPath)
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, int32(100))           // maxMemories
	binary.Write(f, binary.LittleEndian, int32(2_000_000_000)) // 2 billion memories
	f.Close()

	_, err = LoadHippocampus(hipPath)
	if err == nil {
		t.Fatal("expected error for oversized memory count, got nil")
	}
	t.Logf("correctly rejected: %v", err)
}

// TestPersistenceHippocampusNegativeCount verifies LoadHippocampus
// rejects a file with a negative memory count.
func TestPersistenceHippocampusNegativeCount(t *testing.T) {
	dir := t.TempDir()
	hipPath := filepath.Join(dir, "bad.nxhip")

	f, err := os.Create(hipPath)
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, int32(100)) // maxMemories
	binary.Write(f, binary.LittleEndian, int32(-5))  // negative count
	f.Close()

	_, err = LoadHippocampus(hipPath)
	if err == nil {
		t.Fatal("expected error for negative memory count, got nil")
	}
	t.Logf("correctly rejected: %v", err)
}

// TestPersistenceNetworkOversizedNeurons verifies LoadNetwork
// rejects a file with absurdly large neuron count.
func TestPersistenceNetworkOversizedNeurons(t *testing.T) {
	dir := t.TempDir()
	netPath := filepath.Join(dir, "bad.nxnet")

	f, err := os.Create(netPath)
	if err != nil {
		t.Fatal(err)
	}

	hdr := networkFileHeader{}
	copy(hdr.Magic[:], NetworkFileMagic)
	hdr.NeuronCount = 2_000_000_000 // 2 billion neurons
	hdr.SynapseCount = 100
	hdr.HistoryLen = 10

	binary.Write(f, binary.LittleEndian, hdr)
	f.Close()

	_, err = LoadNetwork(netPath, DefaultConfig())
	if err == nil {
		t.Fatal("expected error for oversized neuron count, got nil")
	}
	t.Logf("correctly rejected: %v", err)
}

// TestPersistenceNetworkOversizedSynapses verifies LoadNetwork
// rejects a file with absurdly large synapse count.
func TestPersistenceNetworkOversizedSynapses(t *testing.T) {
	dir := t.TempDir()
	netPath := filepath.Join(dir, "bad.nxnet")

	f, err := os.Create(netPath)
	if err != nil {
		t.Fatal(err)
	}

	hdr := networkFileHeader{}
	copy(hdr.Magic[:], NetworkFileMagic)
	hdr.NeuronCount = 100
	hdr.SynapseCount = 3_000_000_000 // 3 billion synapses
	hdr.HistoryLen = 10

	binary.Write(f, binary.LittleEndian, hdr)
	f.Close()

	_, err = LoadNetwork(netPath, DefaultConfig())
	if err == nil {
		t.Fatal("expected error for oversized synapse count, got nil")
	}
	t.Logf("correctly rejected: %v", err)
}

// TestPersistenceHippocampusOversizedSDR verifies that LoadHippocampus
// rejects a file with an absurdly large SDR size/byte length.
func TestPersistenceHippocampusOversizedSDR(t *testing.T) {
	dir := t.TempDir()
	hipPath := filepath.Join(dir, "bad_sdr.nxhip")

	f, err := os.Create(hipPath)
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, int32(100))       // maxMemories
	binary.Write(f, binary.LittleEndian, int32(1))         // 1 memory
	binary.Write(f, binary.LittleEndian, int32(2_000_000)) // oversized input SDR size
	binary.Write(f, binary.LittleEndian, int32(10))        // input byte length
	f.Close()

	_, err = LoadHippocampus(hipPath)
	if err == nil {
		t.Fatal("expected error for oversized SDR size, got nil")
	}
	t.Logf("correctly rejected: %v", err)
}

func TestPersistenceHippocampusRejectsNegativeContextLength(t *testing.T) {
	dir := t.TempDir()
	hipPath := filepath.Join(dir, "bad_context.nxhip")

	f, err := os.Create(hipPath)
	if err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, int32(100))
	binary.Write(f, binary.LittleEndian, int32(1))
	if err := writeSDR(f, NewSDR(16)); err != nil {
		t.Fatal(err)
	}
	if err := writeSDR(f, NewSDR(16)); err != nil {
		t.Fatal(err)
	}
	binary.Write(f, binary.LittleEndian, uint8(1))
	binary.Write(f, binary.LittleEndian, uint32(0))
	binary.Write(f, binary.LittleEndian, int32(-1))
	f.Close()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoadHippocampus panicked for negative context length: %v", r)
		}
	}()

	_, err = LoadHippocampus(hipPath)
	if err == nil {
		t.Fatal("expected error for negative context length, got nil")
	}
}

func TestPersistenceNetworkRejectsOversizedHistory(t *testing.T) {
	dir := t.TempDir()
	netPath := filepath.Join(dir, "bad_history.nxnet")

	f, err := os.Create(netPath)
	if err != nil {
		t.Fatal(err)
	}

	hdr := networkFileHeader{}
	copy(hdr.Magic[:], NetworkFileMagic)
	hdr.NeuronCount = 10
	hdr.SynapseCount = 0
	hdr.HistoryLen = 2_000_000_000

	binary.Write(f, binary.LittleEndian, hdr)
	f.Close()

	_, err = LoadNetwork(netPath, DefaultConfig())
	if err == nil {
		t.Fatal("expected error for oversized history length, got nil")
	}
}
