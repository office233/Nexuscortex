package cortex

// web_learner.go — Autonomous web knowledge acquisition for Nexus Cortex.
//
// Uses free, no-API-key-required sources:
//   - Wikipedia REST API (summary + full text)
//   - Wiktionary (word definitions)
//
// The WebLearner searches for information, extracts clean text,
// and feeds it through the Organism's learning pipeline.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// WebLearner searches the web and converts discoveries into training data.
type WebLearner struct {
	Client    *http.Client
	RateLimit time.Duration // minimum pause between requests
	lastReq   time.Time

	// Stats
	TotalSearches int
	TotalLearned  int
	TotalFacts    int
}

// NewWebLearner creates a web learner with sensible defaults.
func NewWebLearner() *WebLearner {
	return &WebLearner{
		Client: &http.Client{
			Timeout: 10 * time.Second,
		},
		RateLimit: 2 * time.Second, // Be polite: max 1 req / 2 sec
	}
}

// SearchResult holds one piece of discovered knowledge.
type SearchResult struct {
	Title   string
	Snippet string // Short summary
	Content string // Full clean text
	URL     string
	Source  string // "wikipedia", "wiktionary"
}

// throttle ensures we respect rate limits.
func (wl *WebLearner) throttle() {
	elapsed := time.Since(wl.lastReq)
	if elapsed < wl.RateLimit {
		time.Sleep(wl.RateLimit - elapsed)
	}
	wl.lastReq = time.Now()
}

// SearchWikipedia searches Wikipedia for articles matching the query.
// Returns up to maxResults results with summaries.
func (wl *WebLearner) SearchWikipedia(query string, lang string, maxResults int) ([]SearchResult, error) {
	wl.throttle()
	wl.TotalSearches++

	if lang == "" {
		lang = "en"
	}
	if maxResults <= 0 {
		maxResults = 5
	}

	// Wikipedia search API
	apiURL := fmt.Sprintf("https://%s.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s&srlimit=%d&format=json&utf8=1",
		lang, url.QueryEscape(query), maxResults)

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("wikipedia request build failed: %w", err)
	}
	req.Header.Set("User-Agent", "NexusCortex/1.0 (autonomous learner; contact: nexus@cortex.local)")

	resp, err := wl.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("wikipedia search failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var result struct {
		Query struct {
			Search []struct {
				Title   string `json:"title"`
				Snippet string `json:"snippet"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	var results []SearchResult
	for _, s := range result.Query.Search {
		// Clean HTML tags from snippet
		clean := cleanHTML(s.Snippet)
		results = append(results, SearchResult{
			Title:   s.Title,
			Snippet: clean,
			Source:  "wikipedia-" + lang,
			URL:     fmt.Sprintf("https://%s.wikipedia.org/wiki/%s", lang, url.PathEscape(s.Title)),
		})
	}
	return results, nil
}

// GetWikipediaSummary fetches a concise summary of a Wikipedia article.
func (wl *WebLearner) GetWikipediaSummary(title string, lang string) (*SearchResult, error) {
	wl.throttle()

	if lang == "" {
		lang = "en"
	}

	apiURL := fmt.Sprintf("https://%s.wikipedia.org/api/rest_v1/page/summary/%s",
		lang, url.PathEscape(title))

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("wikipedia summary request failed: %w", err)
	}
	req.Header.Set("User-Agent", "NexusCortex/1.0 (autonomous learner; contact: nexus@cortex.local)")

	resp, err := wl.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("wikipedia summary failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var summary struct {
		Title   string `json:"title"`
		Extract string `json:"extract"`
	}
	if err := json.Unmarshal(body, &summary); err != nil {
		return nil, err
	}

	return &SearchResult{
		Title:   summary.Title,
		Content: summary.Extract,
		Source:  "wikipedia-" + lang,
		URL:     fmt.Sprintf("https://%s.wikipedia.org/wiki/%s", lang, url.PathEscape(title)),
	}, nil
}

// LearnFromResults feeds search results into the Organism's learning pipeline.
func (wl *WebLearner) LearnFromResults(org *Organism, results []SearchResult) int {
	learned := 0
	for _, r := range results {
		text := r.Content
		if text == "" {
			text = r.Snippet
		}
		if text == "" {
			continue
		}

		// Learn the title as a fact
		if r.Title != "" {
			org.LearnQA("What is "+r.Title+"?", text)
			learned++
		}

		// Learn the full text passively
		org.Brain.Learn(text)
		tokens := Tokenize(text)
		org.Wernicke.LearnContext(tokens)

		// Store as episodic memory
		sdr := org.Encoder.EncodeSentence(text)
		org.Hippocampus.Store(sdr, sdr, text)

		learned++
		wl.TotalLearned++
	}
	wl.TotalFacts += learned
	return learned
}

// cleanHTML removes HTML tags from a string (simple approach).
func cleanHTML(s string) string {
	var result strings.Builder
	inTag := false
	for _, r := range s {
		switch {
		case r == '<':
			inTag = true
		case r == '>':
			inTag = false
		case !inTag:
			result.WriteRune(r)
		}
	}
	return strings.TrimSpace(result.String())
}
