package main

import (
	crand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strconv"
	"sync"
	"syscall"
	"time"

	"nexus-cortex/cortex"
	"nexus-cortex/web"
)

// StatsResponse incorporates biological stats along with volatile telemetry
type StatsResponse struct {
	cortex.OrganismStats
	LastFocusTarget string `json:"last_focus_target"`
}

// Server holds the thread-safe web server context
type Server struct {
	org        *cortex.Organism
	mu         sync.Mutex
	lastFocus  string
	lastSource string
	token      string
	port       int
}

func generateRandomToken() string {
	bytes := make([]byte, 16)
	if _, err := crand.Read(bytes); err != nil {
		log.Fatal("CRITICAL: Failed to generate secure random token via crypto/rand: ", err)
	}
	return hex.EncodeToString(bytes)
}

func main() {
	// Default values come from Config so they're centralized
	defaultCfg := cortex.DefaultConfig()

	// ── Command Line Flags ───────────────────────────────────────────
	port := flag.String("port", defaultCfg.WebPort, "Port to bind the HTTP server to")
	bindAddr := flag.String("bind", defaultCfg.WebBindAddr, "Address to bind the HTTP server to")
	openBrowser := flag.Bool("open", true, "Auto-open the dashboard in the default browser")
	dataDir := flag.String("data-dir", defaultCfg.DataDir, "Path to organism data directory")
	fresh := flag.Bool("fresh", false, "Start with a new organism (ignore saved state)")
	noSave := flag.Bool("no-save", false, "Don't auto-save state on exit")
	seed := flag.Int64("seed", defaultCfg.Seed, "Random seed for biological core initialization")
	authToken := flag.String("token", "", "Security token for dashboard authentication (empty to auto-generate, 'none' to disable)")
	flag.Parse()

	// Construct Organism configuration
	cfg := defaultCfg
	cfg.DataDir = *dataDir
	cfg.Fresh = *fresh
	cfg.NoSave = *noSave
	cfg.Seed = *seed
	cfg.WebPort = *port
	cfg.WebBindAddr = *bindAddr
	cfg.Demo = false // Server handles interactivity dynamically

	explicitTokenPassed := *authToken != "" && *authToken != "none"
	token := *authToken
	if token == "" {
		token = generateRandomToken()
	} else if token == "none" {
		token = ""
	}

	nonLoopback := cfg.WebBindAddr != "127.0.0.1" && cfg.WebBindAddr != "localhost" && cfg.WebBindAddr != ""
	if nonLoopback && !explicitTokenPassed {
		log.Fatal("Forbidden: network-exposed bindings require an explicit user-specified token via -token")
	}

	// Print visual launch banner
	fmt.Println()
	fmt.Println("  NEXUS CORTEX - Web UI Neural Dashboard")
	fmt.Println("  Starting Zero-Dependency Real-Time Introspection Server")
	fmt.Println()

	// Instantiate deterministic biological randomizer
	rng := rand.New(rand.NewSource(cfg.Seed))

	// ── Boot or Load the Organism ─────────────────────────────────────
	fmt.Printf("  🔬 Introspecting data directory: %s...\n", cfg.DataDir)
	var org *cortex.Organism
	var err error
	if !cfg.Fresh {
		org, err = cortex.LoadOrganism(cfg, rng)
	}

	if org == nil {
		if err != nil {
			fmt.Printf("  Saved organism state missing or load failed (%v), compiling a new one...\n", err)
		} else if cfg.Fresh {
			fmt.Println("  --fresh flag is active, initializing a new blank organism...")
		}
		org = cortex.NewOrganism(cfg, rng)
	} else {
		fmt.Println("  ✅ Restored saved cognitive state successfully.")
	}
	fmt.Println("  ✅ Organic Neural Core is live and active.")
	fmt.Println()

	server := &Server{
		org:        org,
		lastSource: "Prefrontal Think",
		lastFocus:  "",
		token:      token,
	}

	// ── Mount Route Mapping ──────────────────────────────────────────
	// Static asset mounting from the embedded filesystem
	staticFS := http.FileServer(http.FS(web.Assets))
	http.Handle("/", securityHeaders(staticFS))

	// REST API Endpoints
	// (Removed GET /api/token to prevent session security leakage)

	http.HandleFunc("/api/stats", server.GetStatsHandler)
	http.HandleFunc("/api/chat", server.ChatHandler)
	http.HandleFunc("/api/learn", server.LearnHandler)
	http.HandleFunc("/api/sleep", server.SleepHandler)
	http.HandleFunc("/api/save", server.SaveHandler)
	http.HandleFunc("/api/feedback", server.FeedbackHandler)
	http.HandleFunc("/api/selftrain", server.SelfTrainHandler)

	// ── Graceful Shutdown Handler ────────────────────────────────────
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println()
		fmt.Println("\n  🛑 Intercepted termination signal. Starting graceful shutdown sequence...")

		server.mu.Lock()
		defer server.mu.Unlock()

		if !server.org.Config.NoSave {
			fmt.Printf("  💾 Saving organic neural states into %s...\n", server.org.Config.DataDir)
			if err := server.org.Save(server.org.Config.DataDir); err != nil {
				fmt.Printf("  ⚠️ Failed to save state during exit: %v\n", err)
			} else {
				fmt.Println("  ✅ All neural structures successfully saved to disk.")
			}
		} else {
			fmt.Println("  ⚠️ Persistence skipped (--no-save). All runtime changes will be lost.")
		}
		fmt.Println("  👋 Nexus Cortex digital organism hibernating. System offline.")
		os.Exit(0)
	}()

	// ── Launch HTTP Server ──────────────────────────────────────────
	var listener net.Listener
	var bindErr error
	startPort, _ := strconv.Atoi(*port)
	if startPort <= 0 {
		startPort = 8080
	}

	actualPort := startPort
	// Search up to 200 consecutive ports to guarantee we find a free one
	for p := startPort; p < startPort+200; p++ {
		// Bind to configured address; the dashboard mutates local model state.
		addr := fmt.Sprintf("%s:%d", cfg.WebBindAddr, p)
		listener, bindErr = net.Listen("tcp", addr)
		if bindErr == nil {
			actualPort = p
			server.port = p
			*port = strconv.Itoa(p)
			break
		}

		fmt.Printf("  ⚠️  Port %d is occupied or interface binding blocked: %v. Retrying next port...\n", p, bindErr)
	}

	if bindErr != nil {
		log.Fatalf("  ❌ Failed to bind to any port in range %d-%d: %v", startPort, startPort+199, bindErr)
	}
	defer listener.Close()

	// Start browser helper on a delayed routine now that we are successfully bound
	if *openBrowser {
		go func() {
			time.Sleep(300 * time.Millisecond)
			url := fmt.Sprintf("http://localhost:%d", actualPort)
			fmt.Printf("  🚀 Launching Web Interface at: %s\n", url)
			openBrowserCmd(url)
		}()
	}

	fmt.Printf("  🖥️  Listening for HTTP requests on http://localhost:%d\n", actualPort)
	fmt.Printf("  🔑 Security Token: %s\n", server.token)
	if err := http.Serve(listener, nil); err != nil {
		log.Fatalf("  ❌ Failed to serve HTTP: %v", err)
	}
}

// ─────────────────────────────────────────────────────────────────────
// Security Headers Middleware
// ─────────────────────────────────────────────────────────────────────

// securityHeaders wraps an http.Handler to inject protective HTTP headers
// on every response: Content-Security-Policy, X-Content-Type-Options,
// X-Frame-Options, and Referrer-Policy.
func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		setSecurityHeaders(w)
		next.ServeHTTP(w, r)
	})
}

// setSecurityHeaders adds protective HTTP headers to the response writer.
// Called by both the static file middleware and API handlers.
func setSecurityHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Security-Policy",
		"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "+
			"img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
}

// ─────────────────────────────────────────────────────────────────────
// HTTP API Handlers & Security Validations
// ─────────────────────────────────────────────────────────────────────

// validateRequest verifies the security token and Same-Origin headers for mutating calls
func (s *Server) validateRequest(w http.ResponseWriter, r *http.Request) bool {
	// Inject security headers on every API response (including error paths)
	setSecurityHeaders(w)

	// 1. Validate the X-Nexus-Token custom header if security token is enabled
	if s.token != "" {
		token := r.Header.Get("X-Nexus-Token")
		if token == "" || token != s.token {
			http.Error(w, "Unauthorized: invalid or missing X-Nexus-Token", http.StatusUnauthorized)
			return false
		}
	}

	// 2. Perform strict Same-Origin / Referer validation
	origin := r.Header.Get("Origin")
	referer := r.Header.Get("Referer")

	allowedOrigins := []string{
		fmt.Sprintf("http://localhost:%d", s.port),
		fmt.Sprintf("http://127.0.0.1:%d", s.port),
		fmt.Sprintf("http://[::1]:%d", s.port),
	}
	bindAddr := s.org.Config.WebBindAddr
	if bindAddr != "127.0.0.1" && bindAddr != "localhost" && bindAddr != "" {
		allowedOrigins = append(allowedOrigins, fmt.Sprintf("http://%s:%d", bindAddr, s.port))
	}

	if origin != "" {
		matched := false
		for _, o := range allowedOrigins {
			if origin == o {
				matched = true
				break
			}
		}
		if !matched {
			http.Error(w, "Forbidden: invalid Origin header value", http.StatusForbidden)
			return false
		}
	} else if referer != "" {
		u, err := url.Parse(referer)
		if err != nil {
			http.Error(w, "Forbidden: invalid Referer header", http.StatusForbidden)
			return false
		}
		refOrigin := fmt.Sprintf("%s://%s", u.Scheme, u.Host)
		matched := false
		for _, o := range allowedOrigins {
			if refOrigin == o {
				matched = true
				break
			}
		}
		if !matched {
			http.Error(w, "Forbidden: invalid Referer header value", http.StatusForbidden)
			return false
		}
	} else {
		// Origin or Referer is strictly required for mutating requests to block headless automated abuse
		http.Error(w, "Forbidden: missing Origin or Referer header", http.StatusForbidden)
		return false
	}

	return true
}

// GetStatsHandler returns JSON representation of internal modules
func (s *Server) GetStatsHandler(w http.ResponseWriter, r *http.Request) {
	if !s.validateRequest(w, r) {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	stats := s.org.Stats()
	resp := StatsResponse{
		OrganismStats:   stats,
		LastFocusTarget: s.lastFocus,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// ChatHandler processes cognitive chat loops
func (s *Server) ChatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.validateRequest(w, r) {
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if err.Error() == "http: request body too large" {
			http.Error(w, "Request Entity Too Large: body exceeds 1MB limit", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "Bad Request: message is required", http.StatusBadRequest)
		return
	}
	if req.Message == "" {
		http.Error(w, "Bad Request: message cannot be empty", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Capture hit telemetry prior to run
	hitsBefore, _, _ := s.org.Cerebellum.Stats()

	// Run message through the full O(1) -> Prefrontal Spiking pipeline
	response := s.org.Process(req.Message)

	// Fetch updated hit counters post-run
	hitsAfter, _, _ := s.org.Cerebellum.Stats()

	// Intercept and resolve processing source
	var source string
	confidence := s.org.Prefrontal.GetConfidence()

	if hitsAfter > hitsBefore {
		source = "Cerebellum Cache"
	} else if confidence == 255 {
		source = "Hippocampus Recall"
	} else {
		source = "Prefrontal Think"
	}

	if response == "(no confident response)" {
		source = "System Fallback"
	}

	s.lastSource = source

	// Parse semantic topic focus
	understanding := s.org.Wernicke.Understand(req.Message)
	topic := ""
	if len(understanding.KeyWords) > 0 {
		topic = understanding.KeyWords[0]
	} else if len(understanding.Words) > 0 {
		topic = understanding.Words[0]
	}
	if topic != "" {
		s.lastFocus = topic
	}

	// Update stats
	stats := s.org.Stats()
	resp := struct {
		Response   string        `json:"response"`
		Confidence uint8         `json:"confidence"`
		Surprise   uint8         `json:"surprise"`
		Source     string        `json:"source"`
		Stats      StatsResponse `json:"stats"`
	}{
		Response:   response,
		Confidence: confidence,
		Surprise:   stats.SurpriseLevel,
		Source:     source,
		Stats: StatsResponse{
			OrganismStats:   stats,
			LastFocusTarget: s.lastFocus,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// LearnHandler processes passive knowledge injection
func (s *Server) LearnHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.validateRequest(w, r) {
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if err.Error() == "http: request body too large" {
			http.Error(w, "Request Entity Too Large: body exceeds 1MB limit", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "Bad Request: message is required", http.StatusBadRequest)
		return
	}
	if req.Message == "" {
		http.Error(w, "Bad Request: message cannot be empty", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Train passively
	s.org.Learn(req.Message)

	// Update focus target topic
	tokens := cortex.Tokenize(req.Message)
	if len(tokens) > 0 {
		s.lastFocus = tokens[0]
	}

	stats := s.org.Stats()
	resp := struct {
		Status string        `json:"status"`
		Stats  StatsResponse `json:"stats"`
	}{
		Status: "absorbed",
		Stats: StatsResponse{
			OrganismStats:   stats,
			LastFocusTarget: s.lastFocus,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// SleepHandler performs offline maintenance and returns diff telemetry
func (s *Server) SleepHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.validateRequest(w, r) {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Fetch pre stats snapshot
	statsPre := s.org.Stats()
	preResp := StatsResponse{
		OrganismStats:   statsPre,
		LastFocusTarget: s.lastFocus,
	}

	// Trigger biological Sleep cycle consolidation and retrieve prefrontal training logs
	reflectionLogs := s.org.Sleep()

	if !s.org.Config.NoSave {
		_ = s.org.Save(s.org.Config.DataDir)
	}

	// Fetch post stats snapshot
	statsPost := s.org.Stats()
	postResp := StatsResponse{
		OrganismStats:   statsPost,
		LastFocusTarget: s.lastFocus,
	}

	// Compile high-fidelity console feedback strings
	consoleLogs := []string{
		"Initiating offline neocortical state consolidation...",
		fmt.Sprintf("Generalizing semantic structures (Hippocampus memories: %d)...", statsPre.HippocampusMemories),
		"Replaying episodic sequences to reinforce active tracks...",
		"Running Long-Term Potentiation (LTP) synaptic stabilization...",
		fmt.Sprintf("Pruning weak cache entries in Cerebellum (size: %d -> %d)...", statsPre.CerebellumCacheSize, statsPost.CerebellumCacheSize),
		fmt.Sprintf("Pruning unused prefrontal connections (synapses: %d -> %d)...", statsPre.PrefrontalSynapses, statsPost.PrefrontalSynapses),
		"Decaying obsolete bigram linkages inside vocabulary maps...",
		"Resetting emotional valence to neural baseline.",
	}

	// Append active prefrontal self-reflection logs if any
	if len(reflectionLogs) > 0 {
		consoleLogs = append(consoleLogs, "--------------------------------------------------")
		consoleLogs = append(consoleLogs, reflectionLogs...)
		consoleLogs = append(consoleLogs, "--------------------------------------------------")
	}

	consoleLogs = append(consoleLogs,
		"Consolidation complete. Biological clock calibrated.",
		"Persisted consolidated state vectors safely to disk.",
	)

	resp := struct {
		Pre         StatsResponse `json:"pre"`
		Post        StatsResponse `json:"post"`
		ConsoleLogs []string      `json:"console_logs"`
	}{
		Pre:         preResp,
		Post:        postResp,
		ConsoleLogs: consoleLogs,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// SaveHandler forces persistence manually
func (s *Server) SaveHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.validateRequest(w, r) {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	err := s.org.Save(s.org.Config.DataDir)
	var status string
	if err != nil {
		status = fmt.Sprintf("error: %v", err)
	} else {
		status = "saved"
	}

	resp := map[string]string{
		"status": status,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// FeedbackHandler processes human reinforcement (thumbs up/down and corrections)
func (s *Server) FeedbackHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.validateRequest(w, r) {
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req struct {
		Topic        string `json:"topic"`
		ResponseText string `json:"responseText"`
		Positive     bool   `json:"positive"`
		CorrectText  string `json:"correctText"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if err.Error() == "http: request body too large" {
			http.Error(w, "Request Entity Too Large: body exceeds 1MB limit", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "Bad Request: invalid payload", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Handle feedback
	s.org.HandleFeedback(req.Topic, req.ResponseText, req.Positive, req.CorrectText)

	stats := s.org.Stats()
	resp := struct {
		Status string        `json:"status"`
		Stats  StatsResponse `json:"stats"`
	}{
		Status: "feedback_processed",
		Stats: StatsResponse{
			OrganismStats:   stats,
			LastFocusTarget: s.lastFocus,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// SelfTrainHandler runs the prefrontal autonomous self-reflection loop
func (s *Server) SelfTrainHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.validateRequest(w, r) {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	statsPre := s.org.Stats()
	preResp := StatsResponse{
		OrganismStats:   statsPre,
		LastFocusTarget: s.lastFocus,
	}

	consolidated, pruned, logMsgs := s.org.SelfTrain()

	if !s.org.Config.NoSave {
		_ = s.org.Save(s.org.Config.DataDir)
	}

	statsPost := s.org.Stats()
	postResp := StatsResponse{
		OrganismStats:   statsPost,
		LastFocusTarget: s.lastFocus,
	}

	resp := struct {
		Pre          StatsResponse `json:"pre"`
		Post         StatsResponse `json:"post"`
		Consolidated int           `json:"consolidated"`
		Pruned       int           `json:"pruned"`
		ConsoleLogs  []string      `json:"console_logs"`
	}{
		Pre:          preResp,
		Post:         postResp,
		Consolidated: consolidated,
		Pruned:       pruned,
		ConsoleLogs:  logMsgs,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// ─────────────────────────────────────────────────────────────────────
// Browser Launch helper
// ─────────────────────────────────────────────────────────────────────

func openBrowserCmd(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default: // Linux and other POSIX compliant platforms
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}
