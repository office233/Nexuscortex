// inspect-tokenizer — inspectează BPE tokenizer-ul folosit de cursa D.
// Întrebare: ID-urile speciale 0-4 (PAD, UNK, BOS, EOS, SEP) sunt mapate
// la token-uri speciale așa cum așteaptă codul, sau s-a corupt vocab-ul?
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

type tokenizer struct {
	TokenToID map[string]int `json:"token_to_id"`
	IDToToken map[string]string `json:"id_to_token"`
	VocabSize int            `json:"vocab_size"`
}

func main() {
	path := "data/cortex-auto/tokenizer.json"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(2)
	}
	var t tokenizer
	if err := json.Unmarshal(raw, &t); err != nil {
		fmt.Fprintf(os.Stderr, "unmarshal: %v\n", err)
		os.Exit(2)
	}
	fmt.Printf("vocab_size: %d, token_to_id size: %d, id_to_token size: %d\n",
		t.VocabSize, len(t.TokenToID), len(t.IDToToken))

	// Reverse map: id -> token (handle both possible source maps).
	id2tok := map[int]string{}
	for tok, id := range t.TokenToID {
		id2tok[id] = tok
	}
	for idStr, tok := range t.IDToToken {
		var id int
		fmt.Sscanf(idStr, "%d", &id)
		if _, exists := id2tok[id]; !exists {
			id2tok[id] = tok
		}
	}

	// Print IDs 0..10.
	fmt.Println("\n=== First 11 IDs ===")
	for i := 0; i <= 10; i++ {
		tok, ok := id2tok[i]
		if !ok {
			fmt.Printf("  id=%3d  <missing>\n", i)
		} else {
			fmt.Printf("  id=%3d  %q\n", i, tok)
		}
	}

	// Look for special tokens by name.
	fmt.Println("\n=== Special tokens lookup ===")
	for _, name := range []string{"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"} {
		if id, ok := t.TokenToID[name]; ok {
			fmt.Printf("  %s -> id=%d\n", name, id)
		} else {
			fmt.Printf("  %s -> NOT FOUND\n", name)
		}
	}

	// Sanity: which token is at ID=3 (the configured EOS)?
	fmt.Println("\n=== Critical: token at ID=3 (configured EOSTokenID) ===")
	if tok, ok := id2tok[3]; ok {
		fmt.Printf("  ID 3 -> %q\n", tok)
	}

	// Token "photosynthesis" lookup
	if id, ok := t.TokenToID["photosynthesis"]; ok {
		fmt.Printf("  \"photosynthesis\" -> id=%d (just to confirm it's NOT id 3 here)\n", id)
	} else {
		// Try case variations
		for _, v := range []string{"Photosynthesis", "PHOTOSYNTHESIS", "▁photosynthesis"} {
			if id, ok := t.TokenToID[v]; ok {
				fmt.Printf("  %q -> id=%d\n", v, id)
			}
		}
	}

	// IDs by frequency: print 20 lowest-numbered tokens (sorted).
	fmt.Println("\n=== IDs 0..20 sorted ===")
	ids := []int{}
	for id := range id2tok {
		if id <= 20 {
			ids = append(ids, id)
		}
	}
	sort.Ints(ids)
	for _, id := range ids {
		fmt.Printf("  id=%3d  %q\n", id, id2tok[id])
	}
}
