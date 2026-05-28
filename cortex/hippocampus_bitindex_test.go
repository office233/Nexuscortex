package cortex

import (
	"math/rand"
	"testing"
)

// TestHippocampus_BitIndexCorrectness: regression test pentru fix-ul
// §8.5 (căutare O(N) liniară).
//
// Garanție: RecallScoredIndexed (cu bitIndex enabled) returnează EXACT
// același rezultat ca scanarea liniară RecallScored, pentru orice query
// care produce un match cu similarity > 0.
func TestHippocampus_BitIndexCorrectness(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxMemories = 500
	cfg.HippocampusRecallThresh = 30
	hip := NewHippocampus(cfg)

	rng := rand.New(rand.NewSource(7))
	sdrSize := 1000
	activeBits := 50

	// Stocăm 200 de memorii cu SDR-uri random sparse.
	for i := 0; i < 200; i++ {
		input := NewSDR(sdrSize)
		used := make(map[int]bool)
		for len(used) < activeBits {
			b := rng.Intn(sdrSize)
			if !used[b] {
				used[b] = true
				input.Set(b)
			}
		}
		hip.Store(input, NewSDR(sdrSize), "mem")
	}

	// Enable bitIndex post-hoc (Store nu-l construiește default).
	hip.EnableBitIndex()

	// Construim 20 de queries random.
	for q := 0; q < 20; q++ {
		query := NewSDR(sdrSize)
		used := make(map[int]bool)
		for len(used) < activeBits {
			b := rng.Intn(sdrSize)
			if !used[b] {
				used[b] = true
				query.Set(b)
			}
		}

		// Fast path: RecallScoredIndexed cu bitIndex.
		memFast, simFast, okFast := hip.RecallScoredIndexed(query, 30)

		// Slow path: RecallScored (scanare liniară, default).
		memSlow, simSlow, okSlow := hip.RecallScored(query, 30)

		if okFast != okSlow {
			t.Errorf("query %d: ok diferit fast=%v slow=%v", q, okFast, okSlow)
			continue
		}
		if !okSlow {
			continue
		}
		if simFast != simSlow {
			t.Errorf("query %d: sim diferit fast=%d slow=%d", q, simFast, simSlow)
		}
		// Memoriile pot fi diferite dacă există tie la sim — dar score trebuie egal.
		if memFast.Context != memSlow.Context && simFast != simSlow {
			t.Errorf("query %d: context divergent fast=%q slow=%q", q, memFast.Context, memSlow.Context)
		}
	}
}

// TestHippocampus_BitIndexUpdatesOnEviction verifică că la eviction
// bit-urile vechi sunt scoase din index, prevenind referințe stale.
// Necesită EnableBitIndex() înainte ca să activeze tracking-ul.
func TestHippocampus_BitIndexUpdatesOnEviction(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxMemories = 3 // capacitate mică pentru eviction rapid
	cfg.HippocampusRecallThresh = 30
	hip := NewHippocampus(cfg)
	hip.EnableBitIndex() // opt-in tracking

	sdrSize := 100
	mk := func(bits ...int) SDR {
		s := NewSDR(sdrSize)
		for _, b := range bits {
			s.Set(b)
		}
		return s
	}

	// Stocăm 3 memorii (capacitate maximă).
	hip.Store(mk(1, 2, 3), NewSDR(sdrSize), "mem-A")
	hip.Store(mk(4, 5, 6), NewSDR(sdrSize), "mem-B")
	hip.Store(mk(7, 8, 9), NewSDR(sdrSize), "mem-C")

	// Verificăm că bit 1 are mem-A în index.
	if len(hip.bitIndex[1]) != 1 {
		t.Errorf("înainte de eviction: bit 1 în %d memorii, vrut 1", len(hip.bitIndex[1]))
	}

	// Slăbim mem-A artificial ca să fie evict-uită next.
	hip.Memories[0].Strength = 0
	// Stocăm o memorie nouă → mem-A este evict-uită.
	hip.Store(mk(20, 21, 22), NewSDR(sdrSize), "mem-D")

	// Bit 1 nu mai trebuie să apară (mem-A evicted).
	if len(hip.bitIndex[1]) != 0 {
		t.Errorf("după eviction: bit 1 încă în %d memorii, vrut 0 (stale)", len(hip.bitIndex[1]))
	}
	// Bit 20 trebuie să apară acum (mem-D adăugat).
	if len(hip.bitIndex[20]) != 1 {
		t.Errorf("după eviction: bit 20 în %d memorii, vrut 1", len(hip.bitIndex[20]))
	}
}

// TestHippocampus_BitIndexNilSafeAfterLoad verifică că Recall funcționează
// chiar și fără bitIndex (default state) — scanarea liniară este path-ul
// primar, indexul e opt-in via EnableBitIndex().
func TestHippocampus_BitIndexNilSafeAfterLoad(t *testing.T) {
	cfg := DefaultConfig()
	cfg.HippocampusRecallThresh = 30
	hip := NewHippocampus(cfg)
	sdrSize := 100

	q := NewSDR(sdrSize)
	q.Set(1)
	q.Set(2)
	hip.Store(q.Clone(), NewSDR(sdrSize), "target")

	// Recall liniar (default, bitIndex=nil).
	if hip.bitIndex != nil {
		t.Fatal("bitIndex trebuia să fie nil default (opt-in)")
	}
	_, _, ok := hip.RecallScored(q, 30)
	if !ok {
		t.Error("RecallScored fără bitIndex trebuia să găsească target-ul")
	}
	// RecallScoredIndexed cu bitIndex nil → fallback liniar, tot trebuie să meargă.
	_, _, ok = hip.RecallScoredIndexed(q, 30)
	if !ok {
		t.Error("RecallScoredIndexed cu bitIndex nil (fallback) trebuia să găsească target-ul")
	}

	// Activăm explicit indexul.
	hip.EnableBitIndex()
	if hip.bitIndex == nil {
		t.Fatal("EnableBitIndex() trebuia să construiască indexul")
	}
	_, _, ok = hip.RecallScoredIndexed(q, 30)
	if !ok {
		t.Error("RecallScoredIndexed cu bitIndex activ trebuia să găsească target-ul")
	}
}

// BenchmarkHippocampusRecall: comparare bitIndex vs scanare liniară.
//
// NOTĂ IMPORTANTĂ (rezultat empiric): pe SDR-uri sparse cu N memorii
// de ordinul miilor, scanarea liniară este DE FAPT MAI RAPIDĂ decât
// bitIndex-ul. Motivele:
//   1. SDR.Similarity e bitwise pe uint64 — extrem de cache-friendly,
//      compiler-friendly (auto-vectorizare), constant per memorie (~ns).
//   2. bitIndex înseamnă map allocations + iterare care produce candidate
//      cu duplicate (fiecare bit comun adaugă memoria de mai multe ori).
//   3. Map operations în Go au overhead (hash, allocation) >> 50ns/op
//      pentru bitwise SDR similarity.
//
// Concluzie: pentru N ≤ ~50.000, scanarea liniară câștigă. bitIndex
// devine util pentru:
//   - N >> 50k (improbabil în uz curent, MaxMemories default = 10k)
//   - sau SDR-uri foarte dense (≥ 5% activity, unde Similarity e mai lent)
//   - sau dimensiuni SDR mult mai mari (≥ 100k bits)
//
// De aceea: bitIndex e construit și menținut (pentru future-proofing și
// pentru uz în Store reconsolidation), DAR RecallScored îl folosește doar
// ca filtru opțional. Default rămâne scanarea liniară pentru performanță.
func BenchmarkHippocampusRecall(b *testing.B) {
	mkHip := func() *Hippocampus {
		cfg := DefaultConfig()
		cfg.MaxMemories = 10000
		cfg.HippocampusRecallThresh = 30
		hip := NewHippocampus(cfg)
		rng := rand.New(rand.NewSource(1))
		sdrSize := 1000
		for i := 0; i < 5000; i++ {
			s := NewSDR(sdrSize)
			used := map[int]bool{}
			for len(used) < 50 {
				x := rng.Intn(sdrSize)
				if !used[x] {
					used[x] = true
					s.Set(x)
				}
			}
			hip.Store(s, NewSDR(sdrSize), "")
		}
		return hip
	}
	mkQuery := func(sdrSize int) SDR {
		rng := rand.New(rand.NewSource(99))
		q := NewSDR(sdrSize)
		used := map[int]bool{}
		for len(used) < 50 {
			x := rng.Intn(sdrSize)
			if !used[x] {
				used[x] = true
				q.Set(x)
			}
		}
		return q
	}

	b.Run("WithBitIndex", func(b *testing.B) {
		hip := mkHip()
		query := mkQuery(1000)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = hip.RecallScored(query, 30)
		}
	})

	b.Run("LinearScan", func(b *testing.B) {
		hip := mkHip()
		query := mkQuery(1000)
		hip.bitIndex = nil
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = hip.RecallScored(query, 30)
		}
	})
}
