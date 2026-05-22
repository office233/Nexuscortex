package cortex

// ─────────────────────────────────────────────────────────────────────
// Workspace — Global Workspace (Consciousness)
// ─────────────────────────────────────────────────────────────────────
//
// The Workspace implements a simplified Global Workspace Theory (GWT)
// model of consciousness. Multiple unconscious specialist modules
// submit SDR patterns along with a salience score. The workspace
// selects the most salient pattern for "conscious focus" and
// broadcasts it to all subscribed modules.
//
// Key concepts:
//   - Submit: unconscious modules push candidate patterns.
//   - Focus:  the workspace selects the highest-salience pattern.
//   - Broadcast: the focused pattern is sent to all listeners.
//   - Consciousness: a pattern is "conscious" if it matches
//     the current focus (high overlap).
//
// All arithmetic is integer-only (uint8 salience, overlap counts).

// Default workspace parameters.
const (
	DefaultMaxQueueSize    = 32  // Maximum pending submissions
	DefaultFocusDuration   = 5   // Ticks a pattern stays in focus
	DefaultAttentionThresh = 128 // Minimum salience to enter queue
)

// workspaceEntry pairs an SDR pattern with its salience score.
type workspaceEntry struct {
	Pattern  SDR
	Salience uint8
}

// Workspace is a Global Workspace that selects and broadcasts the
// most salient pattern to subscribed processing modules.
type Workspace struct {
	CurrentFocus    SDR              // Currently focused (conscious) pattern
	BroadcastQueue  []workspaceEntry // Pending pattern submissions
	MaxQueueSize    int              // Maximum queue capacity
	FocusDuration   int              // Ticks remaining for current focus
	AttentionThresh uint8            // Minimum salience to accept submission

	focusTicksLeft int // Countdown for current focus expiry
}

// NewWorkspace creates a Workspace with parameters from Config.
func NewWorkspace(cfg Config) *Workspace {
	maxQueue := cfg.WorkspaceMaxQueueSize
	if maxQueue <= 0 {
		maxQueue = DefaultMaxQueueSize
	}
	focusDur := cfg.WorkspaceFocusDuration
	if focusDur <= 0 {
		focusDur = DefaultFocusDuration
	}
	attThresh := cfg.WorkspaceAttentionThresh
	if attThresh == 0 {
		attThresh = DefaultAttentionThresh
	}
	return &Workspace{
		CurrentFocus:    NewSDR(0),
		BroadcastQueue:  make([]workspaceEntry, 0, maxQueue),
		MaxQueueSize:    maxQueue,
		FocusDuration:   focusDur,
		AttentionThresh: attThresh,
	}
}

// ─────────────────────────────────────────────────────────────────────
// Submit — Push a pattern for consideration
// ─────────────────────────────────────────────────────────────────────

// Submit adds a pattern with the given salience to the broadcast
// queue. Patterns below the attention threshold are ignored.
// If the queue is full, the least salient entry is evicted.
func (w *Workspace) Submit(pattern SDR, salience uint8) {
	if salience < w.AttentionThresh {
		return
	}

	entry := workspaceEntry{
		Pattern:  pattern.Clone(), // Deep copy to prevent aliasing
		Salience: salience,
	}

	if len(w.BroadcastQueue) < w.MaxQueueSize {
		w.BroadcastQueue = append(w.BroadcastQueue, entry)
		return
	}

	// Queue is full — find the least salient entry and replace it
	// if the new entry is more salient.
	minIdx := 0
	minSal := w.BroadcastQueue[0].Salience
	for i := 1; i < len(w.BroadcastQueue); i++ {
		if w.BroadcastQueue[i].Salience < minSal {
			minSal = w.BroadcastQueue[i].Salience
			minIdx = i
		}
	}
	if salience > minSal {
		w.BroadcastQueue[minIdx] = entry
	}
}

// ─────────────────────────────────────────────────────────────────────
// Focus — Select the most salient pattern
// ─────────────────────────────────────────────────────────────────────

// Focus selects the most salient pattern from the broadcast queue,
// sets it as the current focus, and removes it from the queue.
// Returns the focused SDR (empty if the queue is empty).
func (w *Workspace) Focus() SDR {
	if len(w.BroadcastQueue) == 0 {
		return w.CurrentFocus
	}

	// Find the most salient entry.
	bestIdx := 0
	bestSal := w.BroadcastQueue[0].Salience
	for i := 1; i < len(w.BroadcastQueue); i++ {
		if w.BroadcastQueue[i].Salience > bestSal {
			bestSal = w.BroadcastQueue[i].Salience
			bestIdx = i
		}
	}

	w.CurrentFocus = w.BroadcastQueue[bestIdx].Pattern
	w.focusTicksLeft = w.FocusDuration

	// Remove the selected entry by swapping with the last element.
	last := len(w.BroadcastQueue) - 1
	w.BroadcastQueue[bestIdx] = w.BroadcastQueue[last]
	w.BroadcastQueue = w.BroadcastQueue[:last]

	return w.CurrentFocus
}

// ─────────────────────────────────────────────────────────────────────
// Broadcast — Send focused pattern to all listeners
// ─────────────────────────────────────────────────────────────────────

// Broadcast sends the currently focused SDR to all provided module
// functions. Each module receives a copy of the focused pattern for
// independent processing.
func (w *Workspace) Broadcast(modules ...func(SDR)) {
	for _, mod := range modules {
		mod(w.CurrentFocus)
	}
}

// ─────────────────────────────────────────────────────────────────────
// IsConscious — Check if a pattern is currently in focus
// ─────────────────────────────────────────────────────────────────────

// IsConscious returns true if the given pattern has significant
// overlap with the current focus. A pattern is considered conscious
// if at least 50% of its active bits overlap with the focused pattern.
func (w *Workspace) IsConscious(pattern SDR) bool {
	if pattern.ActiveCount == 0 || w.CurrentFocus.ActiveCount == 0 {
		return false
	}

	overlap := pattern.Overlap(w.CurrentFocus)
	// Conscious if overlap ≥ 50% of the pattern's active bits.
	// Integer check: overlap * 2 >= activeCount.
	return overlap*2 >= pattern.ActiveCount
}

// ─────────────────────────────────────────────────────────────────────
// Tick — Advance workspace by one timestep
// ─────────────────────────────────────────────────────────────────────

// Tick advances the workspace by one timestep. If the focus duration
// has expired, it automatically selects the next most salient pattern
// from the queue.
func (w *Workspace) Tick() {
	if w.focusTicksLeft > 0 {
		w.focusTicksLeft--
		return
	}

	// Focus expired — select next pattern if available.
	if len(w.BroadcastQueue) > 0 {
		w.Focus()
	}
}

// ─────────────────────────────────────────────────────────────────────
// Clear — Reset workspace state
// ─────────────────────────────────────────────────────────────────────

// Clear resets the workspace: empties the queue, clears focus, and
// resets the focus countdown.
func (w *Workspace) Clear() {
	w.CurrentFocus = NewSDR(0)
	w.BroadcastQueue = w.BroadcastQueue[:0]
	w.focusTicksLeft = 0
}
