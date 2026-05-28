package cortex

import "testing"

// TestMeasureStability_NoSilentMatchBias: regression test pentru bug-ul
// documentat în HARDCODING_AND_LIMITATIONS.md §8.2.
//
// Bug-ul original folosea accuracy plată (matching bit-cu-bit) într-o
// rețea sparse cu ~5% neuroni activi → 95% dintre neuroni erau silent în
// ambele snapshot-uri și produceau "match" trivial, blocând stability la
// ~95% indiferent de comportamentul real.
//
// Fix-ul corect (Jaccard pe neuroni activi): numără doar neuronii care
// au tras în CEL PUȚIN un snapshot, raportează intersect/union.
//
// Acest test construiește două scenarii care, sub bug-ul vechi, ar fi
// produs ambele ~95%; sub fix-ul corect produc scoruri net diferite.
func TestMeasureStability_NoSilentMatchBias(t *testing.T) {
	netSize := 1000
	activeBits := 50 // 5% sparsity (realistic)

	// Helper: construiește un pattern cu bit-urile date setate la true.
	mkPattern := func(activeIdx ...int) []bool {
		p := make([]bool, netSize)
		for _, i := range activeIdx {
			p[i] = true
		}
		return p
	}

	// Scenariu A: două pattern-uri DISJUNCT active (zero overlap).
	// Stability adevărată = 0 (zero neuroni comuni activi).
	// Sub bug: ~ (netSize - 2*activeBits) / netSize ≈ 90% (silent matches).
	patternA1 := mkPattern(intRange(0, activeBits)...)
	patternA2 := mkPattern(intRange(500, 500+activeBits)...)
	scoreA := measureStability([][]bool{patternA1, patternA2})
	if scoreA > 50 {
		t.Errorf("Scenariu A (disjunct active): stability=%d, așteptat <50 "+
			"(bug-ul de silent-match bias e prezent)", scoreA)
	}

	// Scenariu B: două pattern-uri IDENTICE.
	// Stability adevărată = 255.
	patternB1 := mkPattern(intRange(0, activeBits)...)
	patternB2 := mkPattern(intRange(0, activeBits)...)
	scoreB := measureStability([][]bool{patternB1, patternB2})
	if scoreB != 255 {
		t.Errorf("Scenariu B (identice): stability=%d, așteptat 255", scoreB)
	}

	// Scenariu C: overlap parțial 50% (Jaccard ideal = 50/(50+50-25) = 0.333).
	// Sub fix-ul corect: ≈ 33/100 * 255 ≈ 85.
	patternC1 := mkPattern(intRange(0, activeBits)...)
	patternC2 := mkPattern(intRange(25, 25+activeBits)...) // overlap pe 25 bit-uri
	scoreC := measureStability([][]bool{patternC1, patternC2})
	// Acceptăm un range larg [60, 110] ca să nu fragilizăm testul pe rotunjiri.
	if scoreC < 60 || scoreC > 110 {
		t.Errorf("Scenariu C (overlap 50%%): stability=%d, așteptat în [60,110]", scoreC)
	}

	// Verificare de bază: stab A < stab C < stab B.
	if scoreA >= scoreC {
		t.Errorf("Scenariu A (disjunct, score=%d) trebuie strict < Scenariu C (overlap, score=%d)",
			scoreA, scoreC)
	}
	if scoreC >= scoreB {
		t.Errorf("Scenariu C (overlap, score=%d) trebuie strict < Scenariu B (identic, score=%d)",
			scoreC, scoreB)
	}
}

// TestMeasureStability_EdgeCases acoperă cazurile degenerate.
func TestMeasureStability_EdgeCases(t *testing.T) {
	// Un singur pattern → 0 (nu se poate compara cu nimic).
	if s := measureStability([][]bool{{true, false, true}}); s != 0 {
		t.Errorf("un singur pattern: vrut 0, primit %d", s)
	}
	// Pattern-uri goale (toate silent) → 0 (union=0, skip).
	silent := make([]bool, 100)
	if s := measureStability([][]bool{silent, silent}); s != 0 {
		t.Errorf("toate silent: vrut 0, primit %d (silent-match bias?)", s)
	}
	// Listă goală.
	if s := measureStability(nil); s != 0 {
		t.Errorf("nil: vrut 0, primit %d", s)
	}
}

// intRange returnează slice-ul [start, start+1, ..., end-1].
func intRange(start, end int) []int {
	out := make([]int, 0, end-start)
	for i := start; i < end; i++ {
		out = append(out, i)
	}
	return out
}
