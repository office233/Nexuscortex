package cortex

import (
	"testing"
)

func TestSDRAttentionHeadStoreAndQuery(t *testing.T) {
	head := NewSDRAttentionHead(256, 256, 100)

	// Create some key-value pairs
	key1 := NewSDR(256)
	key1.Set(0)
	key1.Set(1)
	key1.Set(2)
	key1.Set(3)
	key1.Set(4)

	val1 := NewSDR(256)
	val1.Set(100)
	val1.Set(101)
	val1.Set(102)

	key2 := NewSDR(256)
	key2.Set(10)
	key2.Set(11)
	key2.Set(12)
	key2.Set(13)
	key2.Set(14)

	val2 := NewSDR(256)
	val2.Set(200)
	val2.Set(201)
	val2.Set(202)

	head.Store(key1, val1)
	head.Store(key2, val2)

	// Query with something similar to key1
	query := NewSDR(256)
	query.Set(0)
	query.Set(1)
	query.Set(2)
	query.Set(5) // Slightly different
	query.Set(6)

	result := head.Query(query, 1)

	// Should match key1 (3 bits overlap) more than key2 (0 overlap)
	// Result should contain bits from val1
	if result.ActiveCount == 0 {
		t.Error("query returned empty SDR")
	}
}

func TestSDRAttentionNoEntries(t *testing.T) {
	head := NewSDRAttentionHead(256, 256, 100)
	query := NewSDR(256)
	query.Set(0)
	query.Set(1)

	result := head.Query(query, 3)
	if result.ActiveCount != 0 {
		t.Errorf("empty head should return empty SDR, got %d active bits", result.ActiveCount)
	}
}

func TestOverlapCount(t *testing.T) {
	a := NewSDR(128)
	b := NewSDR(128)

	// No overlap
	a.Set(0)
	a.Set(1)
	b.Set(2)
	b.Set(3)

	if overlapCount(a, b) != 0 {
		t.Errorf("expected 0 overlap, got %d", overlapCount(a, b))
	}

	// 1 bit overlap
	b.Set(1)
	if overlapCount(a, b) != 1 {
		t.Errorf("expected 1 overlap, got %d", overlapCount(a, b))
	}

	// 2 bits overlap
	b.Set(0)
	if overlapCount(a, b) != 2 {
		t.Errorf("expected 2 overlap, got %d", overlapCount(a, b))
	}
}

func TestSDRToActivationsRoundtrip(t *testing.T) {
	sdr := NewSDR(64)
	sdr.Set(5)
	sdr.Set(10)
	sdr.Set(63)

	act := sdrToActivations(sdr)
	if len(act) != 64 {
		t.Fatalf("activations length: want 64, got %d", len(act))
	}
	if act[5] != 127 {
		t.Errorf("act[5]: want 127, got %d", act[5])
	}
	if act[10] != 127 {
		t.Errorf("act[10]: want 127, got %d", act[10])
	}
	if act[0] != 0 {
		t.Errorf("act[0]: want 0, got %d", act[0])
	}

	// Convert back
	restored := activationsToSDR(act, 3)
	if restored.ActiveCount != 3 {
		t.Errorf("restored active count: want 3, got %d", restored.ActiveCount)
	}
}

func TestMultiHeadSDRAttention(t *testing.T) {
	mha := NewMultiHeadSDRAttention(128, 64, 4, 50)

	input := NewSDR(128)
	for i := 0; i < 10; i++ {
		input.Set(i * 10)
	}

	// Process multiple tokens
	for i := 0; i < 5; i++ {
		result := mha.ProcessToken(input, 3)
		_ = result // Just verify it doesn't panic
	}
}

func BenchmarkSDRAttentionQuery_100slots(b *testing.B) {
	head := NewSDRAttentionHead(10000, 10000, 100)

	// Fill with random-ish data
	for i := 0; i < 100; i++ {
		key := NewSDR(10000)
		val := NewSDR(10000)
		for j := 0; j < 50; j++ {
			key.Set((i*73 + j*137) % 10000)
			val.Set((i*91 + j*173) % 10000)
		}
		head.Store(key, val)
	}

	query := NewSDR(10000)
	for i := 0; i < 50; i++ {
		query.Set(i * 200)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		head.Query(query, 5)
	}
}

func BenchmarkSDRAttentionQuery_1000slots(b *testing.B) {
	head := NewSDRAttentionHead(10000, 10000, 1000)

	for i := 0; i < 1000; i++ {
		key := NewSDR(10000)
		val := NewSDR(10000)
		for j := 0; j < 50; j++ {
			key.Set((i*73 + j*137) % 10000)
			val.Set((i*91 + j*173) % 10000)
		}
		head.Store(key, val)
	}

	query := NewSDR(10000)
	for i := 0; i < 50; i++ {
		query.Set(i * 200)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		head.Query(query, 5)
	}
}
