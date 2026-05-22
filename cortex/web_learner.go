package cortex

// web_learner.go â€” Autonomous web knowledge acquisition for Nexus Cortex.
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

	// Configurable endpoints and limits (from Config)
	BodyLimit    int64  // max HTTP response body in bytes
	UserAgent    string // User-Agent header
	WikiBaseURL  string // e.g. "wikipedia.org"
	HFSearchURL  string // HuggingFace dataset search URL
	HFRowsURL    string // HuggingFace rows API URL

	// Stats
	TotalSearches int
	TotalLearned  int
	TotalFacts    int
}

// NewWebLearner creates a web learner from Config values.
func NewWebLearner() *WebLearner {
	return NewWebLearnerFromConfig(DefaultConfig())
}

// NewWebLearnerFromConfig creates a web learner with all values from Config.
func NewWebLearnerFromConfig(cfg Config) *WebLearner {
	timeout := time.Duration(cfg.WebLearnerTimeoutSecs) * time.Second
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	rateLimit := time.Duration(cfg.WebLearnerRateLimitMs) * time.Millisecond
	if rateLimit <= 0 {
		rateLimit = 2 * time.Second
	}
	bodyLimitMB := cfg.WebLearnerBodyLimitMB
	if bodyLimitMB <= 0 {
		bodyLimitMB = 5
	}
	userAgent := cfg.WebLearnerUserAgent
	if userAgent == "" {
		userAgent = "NexusCortex/1.0 (autonomous learner)"
	}
	wikiBase := cfg.WebLearnerWikiBaseURL
	if wikiBase == "" {
		wikiBase = "wikipedia.org"
	}
	hfSearch := cfg.WebLearnerHFSearchURL
	if hfSearch == "" {
		hfSearch = "https://huggingface.co/api/datasets"
	}
	hfRows := cfg.WebLearnerHFRowsURL
	if hfRows == "" {
		hfRows = "https://datasets-server.huggingface.co/rows"
	}

	return &WebLearner{
		Client: &http.Client{
			Timeout: timeout,
		},
		RateLimit:   rateLimit,
		BodyLimit:   int64(bodyLimitMB) << 20,
		UserAgent:   userAgent,
		WikiBaseURL: wikiBase,
		HFSearchURL: hfSearch,
		HFRowsURL:   hfRows,
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

func isAllowedURL(targetURL string) bool {
	u, err := url.Parse(targetURL)
	if err != nil {
		return false
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}
	host := strings.ToLower(u.Hostname())
	if host == "" {
		return false
	}

	allowed := []string{"huggingface.co", "datasets-server.huggingface.co", "wikipedia.org"}
	for _, domain := range allowed {
		if host == domain || strings.HasSuffix(host, "."+domain) {
			return true
		}
	}
	return false
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
	apiURL := fmt.Sprintf("https://%s.%s/w/api.php?action=query&list=search&srsearch=%s&srlimit=%d&format=json&utf8=1",
		lang, wl.WikiBaseURL, url.QueryEscape(query), maxResults)

	if !isAllowedURL(apiURL) {
		return nil, fmt.Errorf("SSRF prevention: blocked URL %q", apiURL)
	}

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("wikipedia request build failed: %w", err)
	}
	req.Header.Set("User-Agent", wl.UserAgent)

	resp, err := wl.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("wikipedia search failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("wikipedia search returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, wl.BodyLimit))
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
			URL:     fmt.Sprintf("https://%s.%s/wiki/%s", lang, wl.WikiBaseURL, url.PathEscape(s.Title)),
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

	apiURL := fmt.Sprintf("https://%s.%s/api/rest_v1/page/summary/%s",
		lang, wl.WikiBaseURL, url.PathEscape(title))

	if !isAllowedURL(apiURL) {
		return nil, fmt.Errorf("SSRF prevention: blocked URL %q", apiURL)
	}

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("wikipedia summary request failed: %w", err)
	}
	req.Header.Set("User-Agent", wl.UserAgent)

	resp, err := wl.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("wikipedia summary failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("wikipedia summary returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, wl.BodyLimit))
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
		URL:     fmt.Sprintf("https://%s.%s/wiki/%s", lang, wl.WikiBaseURL, url.PathEscape(title)),
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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// HuggingFace Datasets API integration
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



// SearchHuggingFace searches the HuggingFace Datasets API for datasets
// matching the query. Returns up to maxResults results with dataset
// descriptions as content.
func (wl *WebLearner) SearchHuggingFace(query string, maxResults int) ([]SearchResult, error) {
	wl.throttle()
	wl.TotalSearches++

	if maxResults <= 0 {
		maxResults = 5
	}

	apiURL := fmt.Sprintf("%s?search=%s&limit=%d",
		wl.HFSearchURL, url.QueryEscape(query), maxResults)

	if !isAllowedURL(apiURL) {
		return nil, fmt.Errorf("SSRF prevention: blocked URL %q", apiURL)
	}

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("huggingface request build failed: %w", err)
	}
	req.Header.Set("User-Agent", wl.UserAgent)

	resp, err := wl.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface search failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("huggingface search returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, wl.BodyLimit))
	if err != nil {
		return nil, err
	}

	var datasets []struct {
		ID          string `json:"id"`
		Description string `json:"description"`
		Author      string `json:"author"`
		Downloads   int    `json:"downloads"`
	}
	if err := json.Unmarshal(body, &datasets); err != nil {
		return nil, fmt.Errorf("huggingface parse failed: %w", err)
	}

	var results []SearchResult
	for _, ds := range datasets {
		snippet := ds.Description
		if len(snippet) > 300 {
			snippet = snippet[:300] + "..."
		}
		results = append(results, SearchResult{
			Title:   ds.ID,
			Snippet: snippet,
			Content: ds.Description,
			URL:     "https://huggingface.co/datasets/" + ds.ID,
			Source:  "huggingface",
		})
	}
	return results, nil
}

// LearnFromHuggingFace fetches rows from a HuggingFace dataset and feeds them
// through the Organism's learning pipeline.
// It looks for common field patterns: instruction/output, question/answer,
// text, input/output, etc.
// Returns the count of items successfully learned.
func (wl *WebLearner) LearnFromHuggingFace(org *Organism, datasetID string, maxRows int) (int, error) {
	wl.throttle()

	if maxRows <= 0 {
		maxRows = 20
	}

	apiURL := fmt.Sprintf("%s?dataset=%s&config=default&split=train&offset=0&length=%d",
		wl.HFRowsURL, url.QueryEscape(datasetID), maxRows)

	if !isAllowedURL(apiURL) {
		return 0, fmt.Errorf("SSRF prevention: blocked URL %q", apiURL)
	}

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return 0, fmt.Errorf("huggingface rows request build failed: %w", err)
	}
	req.Header.Set("User-Agent", wl.UserAgent)

	resp, err := wl.Client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("huggingface rows fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("huggingface rows returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, wl.BodyLimit))
	if err != nil {
		return 0, err
	}

	// The rows API returns { "rows": [ { "row": { ... } }, ... ] }
	var result struct {
		Rows []struct {
			Row map[string]interface{} `json:"row"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("huggingface rows parse failed: %w", err)
	}

	learned := 0
	for _, entry := range result.Rows {
		row := entry.Row
		if row == nil {
			continue
		}

		question, answer := extractQAPair(row)

		if question != "" && answer != "" {
			// Instruction/response style â€” use QA learning
			org.LearnQA(question, answer)
			learned++
		} else if text := extractTextField(row); text != "" {
			// Free-text style â€” passive learning
			org.Brain.Learn(text)
			tokens := Tokenize(text)
			org.Wernicke.LearnContext(tokens)
			learned++
		}
	}

	wl.TotalLearned += learned
	wl.TotalFacts += learned
	return learned, nil
}

// extractQAPair tries to find a question/answer pair from a HuggingFace row.
// Supports common field naming conventions.
func extractQAPair(row map[string]interface{}) (question, answer string) {
	// Try instruction/output (Alpaca-style)
	if q, ok := row["instruction"]; ok {
		question = asString(q)
	}
	if a, ok := row["output"]; ok {
		answer = asString(a)
	}
	if question != "" && answer != "" {
		// Append input if present (Alpaca has instruction + input + output)
		if inp, ok := row["input"]; ok {
			if s := asString(inp); s != "" {
				question = question + " " + s
			}
		}
		return
	}

	// Try question/answer (QA-style)
	if q, ok := row["question"]; ok {
		question = asString(q)
	}
	if a, ok := row["answer"]; ok {
		answer = asString(a)
	}
	if question != "" && answer != "" {
		return
	}

	// Try prompt/completion
	if q, ok := row["prompt"]; ok {
		question = asString(q)
	}
	if a, ok := row["completion"]; ok {
		answer = asString(a)
	}
	if question != "" && answer != "" {
		return
	}

	// Try ctx/endings (HellaSwag style)
	if q, ok := row["ctx"]; ok {
		question = asString(q)
	}
	if a, ok := row["endings"]; ok {
		answer = asString(a)
	}
	if question != "" && answer != "" {
		return
	}

	return "", ""
}

// extractTextField tries to find a free-text field from a HuggingFace row.
func extractTextField(row map[string]interface{}) string {
	for _, key := range []string{"text", "content", "sentence", "passage", "document"} {
		if v, ok := row[key]; ok {
			if s := asString(v); s != "" {
				return s
			}
		}
	}
	return ""
}

// asString converts an interface{} to string, handling common JSON types.
func asString(v interface{}) string {
	switch val := v.(type) {
	case string:
		return strings.TrimSpace(val)
	case float64:
		return fmt.Sprintf("%.0f", val)
	case []interface{}:
		// Join array elements (e.g. HellaSwag endings)
		parts := make([]string, 0, len(val))
		for _, item := range val {
			if s, ok := item.(string); ok {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, " | ")
	default:
		if v == nil {
			return ""
		}
		return fmt.Sprintf("%v", v)
	}
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
