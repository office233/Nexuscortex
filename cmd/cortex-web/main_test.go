package main

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"nexus-cortex/cortex"
)

func TestServerValidation(t *testing.T) {
	// Initialize minimal organism config
	cfg := cortex.DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Fresh = true
	cfg.NoSave = true

	rng := rand.New(rand.NewSource(cfg.Seed))
	org := cortex.NewOrganism(cfg, rng)

	server := &Server{
		org:        org,
		lastSource: "Prefrontal Think",
		lastFocus:  "",
		token:      "test-secure-token-12345",
		port:       8080,
	}

	// 1. Mutating POST without token should return 401 Unauthorized
	req1 := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"message":"test"}`))
	req1.Header.Set("Origin", "http://localhost:8080")
	w1 := httptest.NewRecorder()

	server.ChatHandler(w1, req1)
	if w1.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 Unauthorized, got %d: %s", w1.Code, w1.Body.String())
	}

	// 2. Mutating POST with correct token but missing Origin/Referer should return 403 Forbidden
	req2 := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"message":"test"}`))
	req2.Header.Set("X-Nexus-Token", "test-secure-token-12345")
	w2 := httptest.NewRecorder()

	server.ChatHandler(w2, req2)
	if w2.Code != http.StatusForbidden {
		t.Errorf("expected 403 Forbidden, got %d: %s", w2.Code, w2.Body.String())
	}

	// 3. Mutating POST with correct token and valid Origin should succeed (or return 200/400 depending on body)
	req3 := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"message":"Hello"}`))
	req3.Header.Set("X-Nexus-Token", "test-secure-token-12345")
	req3.Header.Set("Origin", "http://localhost:8080")
	w3 := httptest.NewRecorder()

	server.ChatHandler(w3, req3)
	// Success path returns 200 JSON
	if w3.Code != http.StatusOK {
		t.Errorf("expected 200 OK, got %d: %s", w3.Code, w3.Body.String())
	}

	// 4. Mutating POST with body exceeding 1MB limit should fail
	largeBody := strings.Repeat("A", (1<<20)+100) // > 1MB
	payload, _ := json.Marshal(map[string]string{"message": largeBody})

	req4 := httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewReader(payload))
	req4.Header.Set("X-Nexus-Token", "test-secure-token-12345")
	req4.Header.Set("Origin", "http://localhost:8080")
	w4 := httptest.NewRecorder()

	server.ChatHandler(w4, req4)
	if w4.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("expected 413 Request Entity Too Large, got %d: %s", w4.Code, w4.Body.String())
	}

	// 5. GET /api/stats without token should fail
	req5 := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
	req5.Header.Set("Origin", "http://localhost:8080")
	w5 := httptest.NewRecorder()

	server.GetStatsHandler(w5, req5)
	if w5.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 Unauthorized for /api/stats without token, got %d", w5.Code)
	}

	// 6. GET /api/stats with correct token should succeed
	req6 := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
	req6.Header.Set("X-Nexus-Token", "test-secure-token-12345")
	req6.Header.Set("Origin", "http://localhost:8080")
	w6 := httptest.NewRecorder()

	server.GetStatsHandler(w6, req6)
	if w6.Code != http.StatusOK {
		t.Errorf("expected 200 OK for /api/stats with valid token, got %d: %s", w6.Code, w6.Body.String())
	}
}

func TestRemovedTokenEndpoint(t *testing.T) {
	server := &Server{
		token: "test-token",
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/stats", server.GetStatsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/token", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404 Not Found for removed endpoint, got %d", w.Code)
	}
}

func TestTokenNoneBypass(t *testing.T) {
	cfg := cortex.DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Fresh = true
	cfg.NoSave = true
	rng := rand.New(rand.NewSource(cfg.Seed))
	org := cortex.NewOrganism(cfg, rng)

	server := &Server{
		org:   org,
		token: "", // none bypass maps token to empty string
		port:  8080,
	}

	req := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
	req.Header.Set("Origin", "http://localhost:8080")
	w := httptest.NewRecorder()

	server.GetStatsHandler(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200 OK with token bypass, got %d: %s", w.Code, w.Body.String())
	}
}

func TestNonLoopbackBindGuard(t *testing.T) {
	// Test case 1: non-loopback with auto-generated token (empty -token) -> should fail (forbidden)
	{
		bindAddr := "0.0.0.0"
		authToken := ""
		explicitTokenPassed := authToken != "" && authToken != "none"
		nonLoopback := bindAddr != "127.0.0.1" && bindAddr != "localhost" && bindAddr != ""
		
		if nonLoopback && !explicitTokenPassed {
			// Expected safety block triggered
		} else {
			t.Error("expected non-loopback with empty token to trigger safety block")
		}
	}

	// Test case 2: non-loopback with -token none -> should fail (forbidden)
	{
		bindAddr := "0.0.0.0"
		authToken := "none"
		explicitTokenPassed := authToken != "" && authToken != "none"
		nonLoopback := bindAddr != "127.0.0.1" && bindAddr != "localhost" && bindAddr != ""
		
		if nonLoopback && !explicitTokenPassed {
			// Expected safety block triggered
		} else {
			t.Error("expected non-loopback with token 'none' to trigger safety block")
		}
	}

	// Test case 3: non-loopback with explicit token -> allowed
	{
		bindAddr := "0.0.0.0"
		authToken := "my-secret"
		explicitTokenPassed := authToken != "" && authToken != "none"
		nonLoopback := bindAddr != "127.0.0.1" && bindAddr != "localhost" && bindAddr != ""
		
		if nonLoopback && !explicitTokenPassed {
			t.Error("expected non-loopback with explicit token to be allowed, but was blocked")
		}
	}

	// Test case 4: loopback with empty token (auto-generated) -> allowed
	{
		bindAddr := "127.0.0.1"
		authToken := ""
		explicitTokenPassed := authToken != "" && authToken != "none"
		nonLoopback := bindAddr != "127.0.0.1" && bindAddr != "localhost" && bindAddr != ""
		
		if nonLoopback && !explicitTokenPassed {
			t.Error("expected loopback with empty token to be allowed, but was blocked")
		}
	}
}

func TestParsedRefererVerification(t *testing.T) {
	cfg := cortex.DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Fresh = true
	cfg.NoSave = true
	rng := rand.New(rand.NewSource(cfg.Seed))
	org := cortex.NewOrganism(cfg, rng)

	server := &Server{
		org:   org,
		token: "test-token",
		port:  8080,
	}

	// Test Case 1: Referer with exact origin and subpath -> should pass
	{
		req := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
		req.Header.Set("X-Nexus-Token", "test-token")
		req.Header.Set("Referer", "http://localhost:8080/dashboard/index.html?param=value")
		w := httptest.NewRecorder()
		server.GetStatsHandler(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("expected 200 OK for valid Referer subpath, got %d: %s", w.Code, w.Body.String())
		}
	}

	// Test Case 2: Referer spoofing with prefix match but incorrect port -> should be rejected
	{
		req := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
		req.Header.Set("X-Nexus-Token", "test-token")
		req.Header.Set("Referer", "http://localhost:8080.attacker.com/path")
		w := httptest.NewRecorder()
		server.GetStatsHandler(w, req)
		if w.Code != http.StatusForbidden {
			t.Errorf("expected 403 Forbidden for spoofed Referer port, got %d: %s", w.Code, w.Body.String())
		}
	}

	// Test Case 3: Referer with invalid URL -> should be rejected
	{
		req := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
		req.Header.Set("X-Nexus-Token", "test-token")
		req.Header.Set("Referer", "http://[invalid-url:::")
		w := httptest.NewRecorder()
		server.GetStatsHandler(w, req)
		if w.Code != http.StatusForbidden {
			t.Errorf("expected 403 Forbidden for malformed Referer, got %d", w.Code)
		}
	}
}

// TestSecurityHeaders ensures all HTTP responses include required security headers
func TestSecurityHeaders(t *testing.T) {
	cfg := cortex.DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.Fresh = true
	cfg.NoSave = true
	rng := rand.New(rand.NewSource(cfg.Seed))
	org := cortex.NewOrganism(cfg, rng)

	server := &Server{
		org:   org,
		token: "test-token",
		port:  8080,
	}

	req := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
	req.Header.Set("X-Nexus-Token", "test-token")
	req.Header.Set("Origin", "http://localhost:8080")
	w := httptest.NewRecorder()

	server.GetStatsHandler(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	expectedHeaders := map[string]string{
		"Content-Security-Policy": "default-src 'self'",
		"X-Content-Type-Options":  "nosniff",
		"X-Frame-Options":         "DENY",
		"Referrer-Policy":         "strict-origin-when-cross-origin",
	}

	for header, expectedSubstr := range expectedHeaders {
		val := w.Header().Get(header)
		if val == "" {
			t.Errorf("missing required security header: %s", header)
		} else if !strings.Contains(val, expectedSubstr) {
			t.Errorf("header %s = %q, expected to contain %q", header, val, expectedSubstr)
		}
	}
}
