package cortex

import (
	"math/rand"
	"testing"
)

// TestEncodeSentence_ActiveCountConsistency: regression test pentru
// bug-ul de dublă incrementare ActiveCount documentat în
// HARDCODING_AND_LIMITATIONS.md §8.3.
//
// SDR.Set() incrementează deja ActiveCount intern. Orice cod care
// apelează Set() NU trebuie să mai incrementeze manual. Dacă cineva
// adaugă din nou `combined.ActiveCount++` după Set() în EncodeSentence,
// testul prinde regresia.
//
// Verificare: ActiveCount raportat trebuie să fie EXACT egal cu numărul
// real de bit-uri setate la 1 (recount manual).
func TestEncodeSentence_ActiveCountConsistency(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	vocab := NewVocab()
	enc := NewEncoder(vocab, 1000, 50, rng)

	inputs := []string{
		"capitala franței este parisul",
		"câinele meu se numește rex",
		"matematica este frumoasă",
		"a", // single token edge case
		"",  // empty edge case
	}

	for _, text := range inputs {
		sdr := enc.EncodeSentence(text)
		reportedCount := sdr.ActiveCount
		// Recount manual prin parcurgerea bit-urilor.
		actualCount := 0
		for i := 0; i < sdr.Size; i++ {
			if sdr.IsActive(i) {
				actualCount++
			}
		}
		if reportedCount != actualCount {
			t.Errorf("text %q: ActiveCount raportat=%d, real=%d (dublă incrementare?)",
				text, reportedCount, actualCount)
		}
		// ActiveIndices() trebuie să returneze același număr.
		if len(sdr.ActiveIndices()) != actualCount {
			t.Errorf("text %q: ActiveIndices()=%d != actualCount=%d",
				text, len(sdr.ActiveIndices()), actualCount)
		}
	}
}

// TestEncodeWord_ActiveCountMatchesActiveCount verifică același invariant
// pentru EncodeWord (calea simplă, fără shifting). Dacă pe viitor cineva
// modifică EncodeWord, ActiveCount nu trebuie să dublu-numere.
func TestEncodeWord_ActiveCountConsistency(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	vocab := NewVocab()
	enc := NewEncoder(vocab, 1000, 50, rng)

	words := []string{"test", "cuvânt", "x", "encoder", "neuron"}
	for _, w := range words {
		sdr := enc.EncodeWord(w)
		actualCount := 0
		for i := 0; i < sdr.Size; i++ {
			if sdr.IsActive(i) {
				actualCount++
			}
		}
		if sdr.ActiveCount != actualCount {
			t.Errorf("word %q: ActiveCount=%d, real=%d",
				w, sdr.ActiveCount, actualCount)
		}
	}
}
