package cortex

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIsAllowedURL(t *testing.T) {
	tests := []struct {
		url     string
		allowed bool
	}{
		// Valid allowed endpoints (HTTPS-only)
		{"https://wikipedia.org", true},
		{"https://en.wikipedia.org", true},
		{"https://ro.wikipedia.org", true},
		{"https://huggingface.co", true},
		{"https://datasets-server.huggingface.co", true},

		// Blocked because of non-HTTPS scheme
		{"http://wikipedia.org", false},
		{"ftp://wikipedia.org", false},
		{"wikipedia.org", false}, // no scheme

		// Blocked domain spoofing / suffix matching bypass attempts
		{"https://wikipedia.org.evil.com", false},
		{"https://huggingface.co.attacker.net", false},
		{"https://datasets-server.huggingface.co.fake.org", false},

		// Blocked private and local loopback domains/IPs
		{"https://localhost", false},
		{"https://127.0.0.1", false},
		{"https://192.168.1.1", false},
		{"https://10.0.0.1", false},
		{"https://172.16.0.1", false},
		{"https://169.254.169.254", false}, // Cloud Instance Metadata Service
		{"https://[::1]", false},

		// Blocked malformed URLs
		{"https://%gh&", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			res := isAllowedURL(tt.url)
			if res != tt.allowed {
				t.Errorf("expected isAllowedURL(%q) = %v, got %v", tt.url, tt.allowed, res)
			}
		})
	}
}

func TestWebLearnerRedirectSSRF(t *testing.T) {
	// Start a local test server to simulate a redirect hop
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Attempt to redirect to a forbidden address (loopback sensitive data)
		http.Redirect(w, r, "https://127.0.0.1/admin/delete", http.StatusMovedPermanently)
	}))
	defer server.Close()

	wl := NewWebLearner()

	// Build a request to our local redirect server
	req, err := http.NewRequest("GET", server.URL, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Client.Do should follow redirect and fail due to SSRF prevention check
	_, err = wl.Client.Do(req)
	if err == nil {
		t.Fatal("expected request to fail due to forbidden redirect, but it succeeded")
	}

	// Verify the error is due to our SSRF CheckRedirect trigger
	if !strings.Contains(err.Error(), "SSRF prevention") && !strings.Contains(err.Error(), "blocked redirect") {
		t.Errorf("expected SSRF redirect error, got instead: %v", err)
	}
}

func TestRedirectHopLimit(t *testing.T) {
	// Create a chain of 6 redirect hops (exceeds the 5-hop limit)
	hopCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hopCount++
		if hopCount < 8 {
			// Redirect back to ourselves (infinite loop that should be stopped at hop 5)
			http.Redirect(w, r, r.URL.String(), http.StatusFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	wl := NewWebLearner()

	req, err := http.NewRequest("GET", server.URL, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = wl.Client.Do(req)
	if err == nil {
		t.Fatal("expected request to fail due to too many redirects, but it succeeded")
	}

	if !strings.Contains(err.Error(), "too many redirects") && !strings.Contains(err.Error(), "SSRF prevention") {
		t.Errorf("expected redirect hop limit error, got: %v", err)
	}
}
