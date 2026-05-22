package cortex

// ─────────────────────────────────────────────────────────────────────
// Motor Output System — The Body's Output System
// ─────────────────────────────────────────────────────────────────────
//
// Shapes and filters generated text before output. Commands are
// enqueued with a confidence level and only pass through if they
// exceed the current inhibition threshold. Execute pops the
// highest-confidence command from the queue for delivery.

// MotorCommand represents a single output action.
type MotorCommand struct {
	Text       string // Generated text
	Confidence uint8  // How confident in this output
	Urgency    uint8  // How urgent (affects output speed)
	Source     string // Which module generated this
}

// MotorStats exposes high-level metrics for the motor system.
type MotorStats struct {
	QueueDepth      int
	TotalOutputs    uint64
	InhibitionLevel uint8
	HistorySize     int
}

// MotorSystem manages queuing, filtering, and execution of outputs.
type MotorSystem struct {
	OutputQueue     []MotorCommand // Pending outputs
	MaxQueueSize    int
	OutputHistory   []MotorCommand // Past outputs (for learning)
	MaxHistory      int
	TotalOutputs    uint64
	InhibitionLevel uint8 // Higher = more filtering (0=say everything, 255=say nothing)
}

// NewMotorSystem creates a motor system with the given config.
func NewMotorSystem(cfg Config) *MotorSystem {
	maxQueue := cfg.MotorQueueCapacity
	maxHistory := cfg.MotorHistoryCapacity
	return &MotorSystem{
		OutputQueue:   make([]MotorCommand, 0, maxQueue),
		MaxQueueSize:  maxQueue,
		OutputHistory: make([]MotorCommand, 0, maxHistory),
		MaxHistory:    maxHistory,
	}
}

// Enqueue adds a command to the output queue if its confidence
// exceeds the current inhibition level. When the queue is full the
// lowest-confidence command is dropped to make room.
func (m *MotorSystem) Enqueue(cmd MotorCommand) {
	// Gate: confidence must exceed inhibition.
	if cmd.Confidence <= m.InhibitionLevel {
		return
	}

	// If queue is full, drop the lowest-confidence entry.
	if len(m.OutputQueue) >= m.MaxQueueSize {
		minIdx := 0
		for i := 1; i < len(m.OutputQueue); i++ {
			if m.OutputQueue[i].Confidence < m.OutputQueue[minIdx].Confidence {
				minIdx = i
			}
		}
		// Only drop if the new command is better than the weakest.
		if cmd.Confidence <= m.OutputQueue[minIdx].Confidence {
			return
		}
		// Replace the weakest entry.
		m.OutputQueue[minIdx] = cmd
		return
	}

	m.OutputQueue = append(m.OutputQueue, cmd)
}

// Execute pops the highest-confidence command from the queue,
// records it in history, and returns it. Returns false if the
// queue is empty.
func (m *MotorSystem) Execute() (MotorCommand, bool) {
	if len(m.OutputQueue) == 0 {
		return MotorCommand{}, false
	}

	// Find the highest-confidence command.
	bestIdx := 0
	for i := 1; i < len(m.OutputQueue); i++ {
		if m.OutputQueue[i].Confidence > m.OutputQueue[bestIdx].Confidence {
			bestIdx = i
		}
	}

	cmd := m.OutputQueue[bestIdx]

	// Remove from queue (swap with last, truncate).
	last := len(m.OutputQueue) - 1
	m.OutputQueue[bestIdx] = m.OutputQueue[last]
	m.OutputQueue = m.OutputQueue[:last]

	// Record in history (ring buffer to avoid backing array leak).
	if len(m.OutputHistory) >= m.MaxHistory {
		copy(m.OutputHistory, m.OutputHistory[1:])
		m.OutputHistory[len(m.OutputHistory)-1] = cmd
	} else {
		m.OutputHistory = append(m.OutputHistory, cmd)
	}
	m.TotalOutputs++

	return cmd, true
}

// SetInhibition sets the inhibition level.
func (m *MotorSystem) SetInhibition(level uint8) {
	m.InhibitionLevel = level
}

// GetInhibition returns the current inhibition level.
func (m *MotorSystem) GetInhibition() uint8 {
	return m.InhibitionLevel
}

// RecentOutputs returns the last n commands from history.
// If n exceeds history length, all history is returned.
func (m *MotorSystem) RecentOutputs(n int) []MotorCommand {
	if n <= 0 {
		return nil
	}
	h := m.OutputHistory
	if n >= len(h) {
		out := make([]MotorCommand, len(h))
		copy(out, h)
		return out
	}
	out := make([]MotorCommand, n)
	copy(out, h[len(h)-n:])
	return out
}

// Stats returns high-level metrics about the motor system.
func (m *MotorSystem) Stats() MotorStats {
	return MotorStats{
		QueueDepth:      len(m.OutputQueue),
		TotalOutputs:    m.TotalOutputs,
		InhibitionLevel: m.InhibitionLevel,
		HistorySize:     len(m.OutputHistory),
	}
}
