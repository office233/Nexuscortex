package cortex

// ─────────────────────────────────────────────────────────────────────
// Sensory Processing Pipeline — The Body's Input System
// ─────────────────────────────────────────────────────────────────────
//
// Converts raw text input into multi-modal sensory signals. Each input
// is routed through named channels (text, numeric, temporal) that
// detect different properties of the incoming data. The combined SDR
// and per-channel intensities are returned as a SensoryOutput.

// SensoryChannel represents a single sensory modality.
type SensoryChannel struct {
	Name       string // Channel name (e.g., "text", "numeric")
	Pattern    SDR    // Current activation pattern
	Intensity  uint8  // Signal strength (0-255)
	LastUpdate uint64 // Tick of last update
}

// SensoryOutput holds the result of processing a single input.
type SensoryOutput struct {
	Combined    SDR              // Union of all channel patterns
	Channels    map[string]uint8 // Channel name → intensity
	TokenCount  int
	HasNumeric  bool
	HasTemporal bool
}

// SensoryStats exposes high-level metrics for the sensory system.
type SensoryStats struct {
	ChannelCount int
	TotalInputs  uint64
	BufferUsed   int
}

// SensorySystem manages multi-channel sensory processing.
type SensorySystem struct {
	Channels                map[string]*SensoryChannel
	InputBuffer             []SDR // Recent input SDRs (FIFO)
	BufferSize              int
	TotalInputs             uint64
	Encoder                 *Encoder // Shared encoder reference
	TextIntensityScale      int
	NumericIntensityScale   int
	TemporalSequenceScale   int
	TemporalRepetitionScale int
}

// NewSensorySystem creates a sensory system with the given config.
func NewSensorySystem(encoder *Encoder, cfg Config) *SensorySystem {
	bufferSize := cfg.SensoryBufferCapacity
	if bufferSize < 1 {
		bufferSize = 1
	}
	s := &SensorySystem{
		Channels:                make(map[string]*SensoryChannel),
		InputBuffer:             make([]SDR, 0, bufferSize),
		BufferSize:              bufferSize,
		Encoder:                 encoder,
		TextIntensityScale:      cfg.SensoryTextIntensityScale,
		NumericIntensityScale:   cfg.SensoryNumericIntensityScale,
		TemporalSequenceScale:   cfg.SensoryTemporalSequenceScale,
		TemporalRepetitionScale: cfg.SensoryTemporalRepetitionScale,
	}

	// Pre-register default channels.
	s.RegisterChannel("text")
	s.RegisterChannel("numeric")
	s.RegisterChannel("temporal")

	return s
}

// RegisterChannel creates and returns a new sensory channel.
// If the channel already exists, the existing one is returned.
func (s *SensorySystem) RegisterChannel(name string) *SensoryChannel {
	if ch, ok := s.Channels[name]; ok {
		return ch
	}
	ch := &SensoryChannel{
		Name: name,
	}
	s.Channels[name] = ch
	return ch
}

// GetChannel returns the named channel, or nil if it doesn't exist.
func (s *SensorySystem) GetChannel(name string) *SensoryChannel {
	return s.Channels[name]
}

// ProcessInput tokenizes and encodes raw text, activating the
// appropriate sensory channels based on content analysis.
func (s *SensorySystem) ProcessInput(text string) SensoryOutput {
	tokens := Tokenize(text)

	// Encode every token into an SDR and union them together.
	tokenSDRs := s.Encoder.EncodeText(text)

	var combined SDR
	if len(tokenSDRs) > 0 {
		combined = tokenSDRs[0]
		for i := 1; i < len(tokenSDRs); i++ {
			combined = combined.Union(tokenSDRs[i])
		}
	} else {
		combined = NewSDR(s.Encoder.sdrSize)
	}

	// Push into the input buffer (ring buffer to avoid backing array leak).
	if len(s.InputBuffer) >= s.BufferSize {
		copy(s.InputBuffer, s.InputBuffer[1:])
		s.InputBuffer[len(s.InputBuffer)-1] = combined
	} else {
		s.InputBuffer = append(s.InputBuffer, combined)
	}
	s.TotalInputs++

	// ── Channel activation ──────────────────────────────────────

	channels := make(map[string]uint8)

	// Text channel — always active when tokens are present.
	textIntensity := uint8(0)
	if len(tokens) > 0 {
		// Scale intensity by token count, capped at 255.
		ti := len(tokens) * s.TextIntensityScale
		if ti > 255 {
			ti = 255
		}
		textIntensity = uint8(ti)
	}
	s.activateChannel("text", combined, textIntensity)
	channels["text"] = textIntensity

	// Numeric channel — activated when any token is all digits.
	hasNumeric := false
	numericCount := 0
	for _, tok := range tokens {
		if isAllDigits(tok) {
			hasNumeric = true
			numericCount++
		}
	}
	numericIntensity := uint8(0)
	if hasNumeric {
		ni := numericCount * s.NumericIntensityScale
		if ni > 255 {
			ni = 255
		}
		numericIntensity = uint8(ni)
		s.activateChannel("numeric", combined, numericIntensity)
	}
	channels["numeric"] = numericIntensity

	// Temporal channel — activated by sequence length and word repetition.
	hasTemporal := false
	temporalIntensity := uint8(0)
	if len(tokens) > 0 {
		// Detect repetition: count unique tokens vs total (integer math).
		unique := make(map[string]struct{}, len(tokens))
		for _, tok := range tokens {
			unique[tok] = struct{}{}
		}

		// Long sequences or high repetition trigger temporal awareness.
		seqScore := len(tokens) * s.TemporalSequenceScale
		repScore := 0
		if len(tokens) > 0 {
			repScore = (len(tokens) - len(unique)) * s.TemporalRepetitionScale / len(tokens)
		}
		total := seqScore + repScore
		if total > 255 {
			total = 255
		}
		if total > 0 {
			hasTemporal = true
			temporalIntensity = uint8(total)
			s.activateChannel("temporal", combined, temporalIntensity)
		}
	}
	channels["temporal"] = temporalIntensity

	return SensoryOutput{
		Combined:    combined,
		Channels:    channels,
		TokenCount:  len(tokens),
		HasNumeric:  hasNumeric,
		HasTemporal: hasTemporal,
	}
}

// activateChannel updates a channel's pattern and intensity.
func (s *SensorySystem) activateChannel(name string, pattern SDR, intensity uint8) {
	ch := s.Channels[name]
	if ch == nil {
		ch = s.RegisterChannel(name)
	}
	ch.Pattern = pattern.Clone() // Deep copy to prevent aliasing
	ch.Intensity = intensity
	ch.LastUpdate = s.TotalInputs
}

// isAllDigits returns true if every rune in s is an ASCII digit.
func isAllDigits(s string) bool {
	if len(s) == 0 {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// RecentPatterns returns the SDRs currently in the input buffer.
func (s *SensorySystem) RecentPatterns() []SDR {
	out := make([]SDR, len(s.InputBuffer))
	copy(out, s.InputBuffer)
	return out
}

// Stats returns high-level metrics about the sensory system.
func (s *SensorySystem) Stats() SensoryStats {
	return SensoryStats{
		ChannelCount: len(s.Channels),
		TotalInputs:  s.TotalInputs,
		BufferUsed:   len(s.InputBuffer),
	}
}
