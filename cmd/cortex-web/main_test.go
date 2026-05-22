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
