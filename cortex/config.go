package cortex

import (
	"fmt"
	"strconv"
)

// Config represents all configurations, sizes, thresholds, seeds, paths,
// and learning knobs for the Nexus Cortex digital organism.
type Config struct {
	DataDir      string `json:"data_dir"`
	Seed         int64  `json:"seed"`
	Demo         bool   `json:"demo"`
	NoSave       bool   `json:"no_save"`
	Fresh        bool   `json:"fresh"`
	MaxGenWords  int    `json:"max_gen_words"`
	SDRSize      int    `json:"sdr_size"`
	ActiveCount  int    `json:"active_count"`
	MaxMemories  int    `json:"max_memories"`
	ThinkCycles  int    `json:"think_cycles"`

	CerebellumConfThreshold uint8  `json:"cerebellum_conf_threshold"`
	HippocampusRecallThresh uint8  `json:"hippocampus_recall_thresh"`
	PrefrontalConfThreshold uint8  `json:"prefrontal_conf_threshold"`
	BrainPruneMaxAge        uint8  `json:"brain_prune_max_age"`
	CerebellumMinUseCount   uint32 `json:"cerebellum_min_use_count"`
	PrefrontalPruneThresh   uint8  `json:"prefrontal_prune_thresh"`

	// Thousand Brains configuration
	ThousandBrainsColumns        int     `json:"thousand_brains_columns"`
	ThousandBrainsColNeurons     int     `json:"thousand_brains_col_neurons"`
	ThousandBrainsColConnectivity float64 `json:"thousand_brains_col_connectivity"`
	ThousandBrainsProcessTicks   int     `json:"thousand_brains_process_ticks"`
	ThousandBrainsInputCurrent   uint8   `json:"thousand_brains_input_current"`

	// Reward System configuration
	RewardSystemCapacity int `json:"reward_system_capacity"`

	// Emotion Engine configuration
	EmotionHistoryCapacity  int   `json:"emotion_history_capacity"`
	EmotionMomentumDecay    uint8 `json:"emotion_momentum_decay"`
	EmotionValenceWeight    uint8 `json:"emotion_valence_weight"`
	EmotionArousalWeight    uint8 `json:"emotion_arousal_weight"`
	EmotionCuriosityWeight  uint8 `json:"emotion_curiosity_weight"`
	EmotionConfidenceWeight uint8 `json:"emotion_confidence_weight"`
	EmotionSocialWeight     uint8 `json:"emotion_social_weight"`

	// Curiosity Drive configuration
	CuriosityHistoryCapacity int `json:"curiosity_history_capacity"`

	// Sensory System configuration
	SensoryBufferCapacity          int `json:"sensory_buffer_capacity"`
	SensoryTextIntensityScale      int `json:"sensory_text_intensity_scale"`
	SensoryNumericIntensityScale   int `json:"sensory_numeric_intensity_scale"`
	SensoryTemporalSequenceScale   int `json:"sensory_temporal_sequence_scale"`
	SensoryTemporalRepetitionScale int `json:"sensory_temporal_repetition_scale"`

	// Motor System configuration
	MotorQueueCapacity   int `json:"motor_queue_capacity"`
	MotorHistoryCapacity int `json:"motor_history_capacity"`

	// Cognitive Workspace configuration
	MinAttentionSalience int `json:"min_attention_salience"`

	// Prefrontal Reasoning configuration
	PrefrontalNetSize            int     `json:"prefrontal_net_size"`
	PrefrontalConnectivity       float64 `json:"prefrontal_connectivity"`
	PrefrontalInputCurrent       uint8   `json:"prefrontal_input_current"`
	PrefrontalMaxHops            int     `json:"prefrontal_max_hops"`
	PrefrontalConvergenceThresh  uint8   `json:"prefrontal_convergence_thresh"`

	// Encoder configuration
	EncoderContextMaxSize int `json:"encoder_context_max_size"`

	// Self-Model threshold configuration
	SelfKnowsAboutThreshold  uint32 `json:"self_knows_about_threshold"`
	SelfWeakTopicThreshold   uint32 `json:"self_weak_topic_threshold"`
	SelfStrongTopicThreshold uint32 `json:"self_strong_topic_threshold"`

	// Network & LIF Neuron configuration
	NetworkExcitatoryRatioNumerator int   `json:"network_excitatory_ratio_numerator"` // e.g. 4 for 4/5 (80%)
	NeuronMinThreshold              uint8 `json:"neuron_min_threshold"`
	NeuronMaxThreshold              uint8 `json:"neuron_max_threshold"`
	NeuronMinLeak                   uint8 `json:"neuron_min_leak"`
	NeuronMaxLeak                   uint8 `json:"neuron_max_leak"`
	SynapseMinWeight                uint8 `json:"synapse_min_weight"`
	SynapseMaxWeight                uint8 `json:"synapse_max_weight"`
	SynapseMinDelay                 uint8 `json:"synapse_min_delay"`
	SynapseMaxDelay                 uint8 `json:"synapse_max_delay"`

	// Hippocampus (Episodic Memory) configuration
	HippoReconsolidationThresh uint8 `json:"hippo_reconsolidation_thresh"`
	HippoInitialStrength       uint8 `json:"hippo_initial_strength"`
	HippoLtpThreshold          uint8 `json:"hippo_ltp_threshold"`

	// Broca Language Generation configuration
	BrocaConfidenceProximity   uint8  `json:"broca_confidence_proximity"`
	BrocaTopNCandidates        int    `json:"broca_top_n_candidates"`
	BrocaSequentialMultiplier  uint16 `json:"broca_sequential_multiplier"`

	// Rhythm Engine configuration
	RhythmSleepThreshold uint8 `json:"rhythm_sleep_threshold"`

	// Wernicke Language Comprehension configuration
	WernickeKeywordThreshold uint32 `json:"wernicke_keyword_threshold"`

	// Sequence Memory configuration
	SequenceMemoryMaxTargets      int    `json:"sequence_memory_max_targets"`
	SequenceMemoryIncrementWeight uint16 `json:"sequence_memory_increment_weight"`

	// STDP Learning configuration
	StdpPotentiate uint8 `json:"stdp_potentiate"`
	StdpDepress    uint8 `json:"stdp_depress"`
	StdpTraceDecay uint8 `json:"stdp_trace_decay"`
	StdpMaxWeight  uint8 `json:"stdp_max_weight"`

	// Neuron Dynamics configuration
	NeuronRefractoryPeriod    uint8 `json:"neuron_refractory_period"`
	HomeostaticSpikeIncrement uint8 `json:"homeostatic_spike_increment"`
	HomeostaticBaselineFloor  uint8 `json:"homeostatic_baseline_floor"`
	HomeostaticDecayRate      uint8 `json:"homeostatic_decay_rate"`

	// WTA Columnar Inhibition configuration
	WtaColumnSize        int    `json:"wta_column_size"`
	WtaTraceIncrement    uint32 `json:"wta_trace_increment"`
	WtaActivationThresh  uint32 `json:"wta_activation_thresh"`
	WtaMaxInhibition     uint16 `json:"wta_max_inhibition"`

	// Noradrenergic Reset configuration
	NoradrenergicChaosThresholdPct int    `json:"noradrenergic_chaos_threshold_pct"`
	NoradrenergicChaosCounterLimit int    `json:"noradrenergic_chaos_counter_limit"`
	NoradrenergicThresholdBoost    uint8  `json:"noradrenergic_threshold_boost"`
	NoradrenergicDampeningTicks    uint16 `json:"noradrenergic_dampening_ticks"`
	NoradrenergicDampeningBias     uint16 `json:"noradrenergic_dampening_bias"`
	NoradrenergicCooldownPct       int    `json:"noradrenergic_cooldown_pct"`

	// Brain Learning Weight configuration
	BrainBigramWeight      uint16 `json:"brain_bigram_weight"`
	BrainSkipgramWeight    uint16 `json:"brain_skipgram_weight"`
	BrainSemanticWeight    uint16 `json:"brain_semantic_weight"`
	BrainContextWindowSize int    `json:"brain_context_window_size"`
	FeedbackLTPAmount      uint16 `json:"feedback_ltp_amount"`
	FeedbackLTDAmount      uint16 `json:"feedback_ltd_amount"`
	FeedbackNewSynapseWeight uint16 `json:"feedback_new_synapse_weight"`

	// Broca Generation configuration
	BrocaDecodeTopK int `json:"broca_decode_top_k"`

	// Self-Training configuration
	SelfTrainMaxTopics    int   `json:"self_train_max_topics"`
	SelfTrainRecallThresh uint8 `json:"self_train_recall_thresh"`
	SelfTrainStableConf   uint8 `json:"self_train_stable_conf"`
	SelfTrainUnstableConf uint8 `json:"self_train_unstable_conf"`

	// Curiosity Dynamics configuration
	CuriosityBoredThreshold    uint8 `json:"curiosity_bored_threshold"`
	CuriosityOverwhelmThreshold uint8 `json:"curiosity_overwhelm_threshold"`
	CuriosityInterestIncrement uint8 `json:"curiosity_interest_increment"`
	CuriosityInterestDecay     uint8 `json:"curiosity_interest_decay"`
	CuriosityRateStep          uint8 `json:"curiosity_rate_step"`

	// Emotion Dynamics configuration
	EmotionStabilityThreshold int `json:"emotion_stability_threshold"`
	EmotionConfidenceBaseline uint8 `json:"emotion_confidence_baseline"`

	// Reward Curve configuration
	RewardSweetSpotLow  uint8 `json:"reward_sweet_spot_low"`
	RewardSweetSpotHigh uint8 `json:"reward_sweet_spot_high"`
	RewardLowValue      int8  `json:"reward_low_value"`

	// Predictor configuration
	PredictorWindowSize  int `json:"predictor_window_size"`
	PredictorUnionDepth  int `json:"predictor_union_depth"`
	PredictorMaxHistory  int `json:"predictor_max_history"`

	// Workspace (consciousness) configuration
	WorkspaceMaxQueueSize    int   `json:"workspace_max_queue_size"`
	WorkspaceFocusDuration   int   `json:"workspace_focus_duration"`
	WorkspaceAttentionThresh uint8 `json:"workspace_attention_thresh"`

	// Cerebellum configuration
	CerebellumMaxCacheSize int `json:"cerebellum_max_cache_size"`

	// Working Memory configuration
	WorkingMemoryCapacity     int   `json:"working_memory_capacity"`
	WorkingMemoryDecayRate    uint8 `json:"working_memory_decay_rate"`
	WorkingMemoryMinRelevance uint8 `json:"working_memory_min_relevance"`

	// Beam Search configuration
	BeamSearchWidth                int `json:"beam_search_width"`
	BeamSearchMaxCandidatesPerStep int `json:"beam_search_max_candidates_per_step"`

	// Error-Driven Learning configuration
	ErrorLearningStrength  uint8 `json:"error_learning_strength"`  // Max synapse adjustment per call
	ErrorLearningThreshold uint8 `json:"error_learning_threshold"` // Min prediction error to trigger

	// Attention Module configuration
	AttentionHistorySize   int   `json:"attention_history_size"`    // Ring buffer capacity for context history
	AttentionMinWeight     uint8 `json:"attention_min_weight"`      // Minimum weight to keep a bit active
	AttentionContextBoost  uint8 `json:"attention_context_boost"`   // Bonus weight for bits active in current context
	AttentionFrequencyScale uint8 `json:"attention_frequency_scale"` // Weight per historical occurrence

	// Analogy Engine configuration
	AnalogyMaxCandidates int `json:"analogy_max_candidates"` // Max vocab words to scan during analogy search

	// Sleep Consolidation configuration
	SleepReplayCount     int   `json:"sleep_replay_count"`     // Memories replayed per sleep cycle
	SleepInterleaveRatio int   `json:"sleep_interleave_ratio"` // Old memories per new memory during replay
	SleepStabilityThresh uint8 `json:"sleep_stability_thresh"` // Min prefrontal stability to reinforce

	// Autonomous Learning configuration — all previously hardcoded
	AutoSeedTopics   []string `json:"auto_seed_topics,omitempty"`   // Initial curiosity topics
	AutoSeedDatasets []string `json:"auto_seed_datasets,omitempty"` // HuggingFace dataset IDs
	AutoSearchLangs  []string `json:"auto_search_langs,omitempty"`  // Wikipedia search languages
	AutoHFRowsPerDS  int      `json:"auto_hf_rows_per_ds"`         // Max rows per HF dataset
	AutoLearnInterval int     `json:"auto_learn_interval_secs"`    // Seconds between learn cycles
	AutoMaxGapsPerCycle int   `json:"auto_max_gaps_per_cycle"`     // Max gaps to address per cycle

	// Web server configuration
	WebPort     string `json:"web_port,omitempty"`      // Dashboard port (default "8080")
	WebBindAddr string `json:"web_bind_addr,omitempty"` // Bind address (default "127.0.0.1")

	// WebLearner HTTP configuration — previously hardcoded
	WebLearnerTimeoutSecs  int    `json:"web_learner_timeout_secs"`  // HTTP request timeout
	WebLearnerRateLimitMs  int    `json:"web_learner_rate_limit_ms"` // Min pause between requests (ms)
	WebLearnerBodyLimitMB  int    `json:"web_learner_body_limit_mb"` // Max HTTP response body (MB)
	WebLearnerUserAgent    string `json:"web_learner_user_agent,omitempty"`
	WebLearnerWikiBaseURL  string `json:"web_learner_wiki_base_url,omitempty"`  // e.g. "wikipedia.org"
	WebLearnerHFSearchURL  string `json:"web_learner_hf_search_url,omitempty"`
	WebLearnerHFRowsURL    string `json:"web_learner_hf_rows_url,omitempty"`
}

// DefaultConfig returns a configuration with sensible biological and cognitive defaults.
func DefaultConfig() Config {
	return Config{
		DataDir:                 "./data/cortex",
		Seed:                    42,
		Demo:                    true,
		NoSave:                  false,
		Fresh:                   false,
		MaxGenWords:             20,
		SDRSize:                 10000,
		ActiveCount:             50,
		MaxMemories:             10000,
		ThinkCycles:             10,
		CerebellumConfThreshold: 178,
		HippocampusRecallThresh: 180,
		PrefrontalConfThreshold: 128,
		BrainPruneMaxAge:        200,
		CerebellumMinUseCount:   2,
		PrefrontalPruneThresh:   5,

		// Thousand Brains defaults
		ThousandBrainsColumns:        64,
		ThousandBrainsColNeurons:     100,
		ThousandBrainsColConnectivity: 0.10,
		ThousandBrainsProcessTicks:   5,
		ThousandBrainsInputCurrent:   200,

		// Reward System defaults
		RewardSystemCapacity: 1000,

		// Emotion Engine defaults
		EmotionHistoryCapacity:  100,
		EmotionMomentumDecay:    10,
		EmotionValenceWeight:    70,
		EmotionArousalWeight:    40,
		EmotionCuriosityWeight:  35,
		EmotionConfidenceWeight: 20,
		EmotionSocialWeight:     25,

		// Curiosity Drive defaults
		CuriosityHistoryCapacity: 500,

		// Sensory System defaults
		SensoryBufferCapacity:          100,
		SensoryTextIntensityScale:      32,
		SensoryNumericIntensityScale:   64,
		SensoryTemporalSequenceScale:   8,
		SensoryTemporalRepetitionScale: 128,

		// Motor System defaults
		MotorQueueCapacity:   10,
		MotorHistoryCapacity: 1000,

		// Cognitive Workspace defaults
		MinAttentionSalience: 50,

		// Prefrontal Reasoning defaults
		PrefrontalNetSize:            3000,
		PrefrontalConnectivity:       0.05,
		PrefrontalInputCurrent:       128,
		PrefrontalMaxHops:            3,
		PrefrontalConvergenceThresh:  240,

		// Encoder defaults
		EncoderContextMaxSize: 10,

		// Self-Model threshold defaults
		SelfKnowsAboutThreshold:  50,
		SelfWeakTopicThreshold:   30,
		SelfStrongTopicThreshold: 200,

		// Network & LIF Neuron defaults
		NetworkExcitatoryRatioNumerator: 4,
		NeuronMinThreshold:              100,
		NeuronMaxThreshold:              200,
		NeuronMinLeak:                   1,
		NeuronMaxLeak:                   4,
		SynapseMinWeight:                10,
		SynapseMaxWeight:                80,
		SynapseMinDelay:                 1,
		SynapseMaxDelay:                 4,

		// Hippocampus (Episodic Memory) defaults
		HippoReconsolidationThresh: 204,
		HippoInitialStrength:       10,
		HippoLtpThreshold:          128,

		// Broca Language Generation defaults
		BrocaConfidenceProximity:   15,
		BrocaTopNCandidates:        5,
		BrocaSequentialMultiplier:  3,

		// Rhythm Engine defaults
		RhythmSleepThreshold: 200,

		// Wernicke Language Comprehension defaults
		WernickeKeywordThreshold: 2,

		// Sequence Memory defaults
		SequenceMemoryMaxTargets:      64,
		SequenceMemoryIncrementWeight: 10,

		// STDP Learning defaults
		StdpPotentiate: 10,
		StdpDepress:    6,
		StdpTraceDecay: 20,
		StdpMaxWeight:  250,

		// Neuron Dynamics defaults
		NeuronRefractoryPeriod:    15,
		HomeostaticSpikeIncrement: 8,
		HomeostaticBaselineFloor:  80,
		HomeostaticDecayRate:      1,

		// WTA Columnar Inhibition defaults
		WtaColumnSize:        100,
		WtaTraceIncrement:    256,
		WtaActivationThresh:  50,
		WtaMaxInhibition:     100,

		// Noradrenergic Reset defaults
		NoradrenergicChaosThresholdPct: 30,
		NoradrenergicChaosCounterLimit: 8,
		NoradrenergicThresholdBoost:    30,
		NoradrenergicDampeningTicks:    10,
		NoradrenergicDampeningBias:     15,
		NoradrenergicCooldownPct:       5,

		// Brain Learning Weight defaults
		BrainBigramWeight:        10,
		BrainSkipgramWeight:      5,
		BrainSemanticWeight:      2,
		BrainContextWindowSize:   5,
		FeedbackLTPAmount:        50,
		FeedbackLTDAmount:        40,
		FeedbackNewSynapseWeight: 50,

		// Broca Generation defaults
		BrocaDecodeTopK: 50,

		// Self-Training defaults
		SelfTrainMaxTopics:    5,
		SelfTrainRecallThresh: 50,
		SelfTrainStableConf:   200,
		SelfTrainUnstableConf: 100,

		// Curiosity Dynamics defaults
		CuriosityBoredThreshold:    40,
		CuriosityOverwhelmThreshold: 200,
		CuriosityInterestIncrement: 10,
		CuriosityInterestDecay:     5,
		CuriosityRateStep:          10,

		// Emotion Dynamics defaults
		EmotionStabilityThreshold: 30,
		EmotionConfidenceBaseline: 128,

		// Reward Curve defaults
		RewardSweetSpotLow:  30,
		RewardSweetSpotHigh: 100,
		RewardLowValue:      10,

		// Predictor defaults
		PredictorWindowSize: 5,
		PredictorUnionDepth: 3,
		PredictorMaxHistory: 100,

		// Workspace defaults
		WorkspaceMaxQueueSize:    32,
		WorkspaceFocusDuration:   5,
		WorkspaceAttentionThresh: 128,

		// Cerebellum defaults
		CerebellumMaxCacheSize: 10000,

		// Working Memory defaults
		WorkingMemoryCapacity:     8,
		WorkingMemoryDecayRate:    3,
		WorkingMemoryMinRelevance: 10,

		// Beam Search defaults
		BeamSearchWidth:                5,
		BeamSearchMaxCandidatesPerStep: 10,

		// Error-Driven Learning defaults
		ErrorLearningStrength:  5,
		ErrorLearningThreshold: 100,

		// Attention Module defaults
		AttentionHistorySize:   10,
		AttentionMinWeight:     64,
		AttentionContextBoost:  100,
		AttentionFrequencyScale: 25,

		// Analogy Engine defaults
		AnalogyMaxCandidates: 100,

		// Sleep Consolidation defaults
		SleepReplayCount:     10,
		SleepInterleaveRatio: 2,
		SleepStabilityThresh: 180,

		// Autonomous Learning defaults (previously hardcoded in constructor)
		AutoSeedTopics: []string{
			"photosynthesis", "DNA", "evolution", "gravity", "atom",
			"quantum mechanics", "relativity", "cell biology",
			"algebra", "geometry", "calculus", "probability",
			"prime number", "fibonacci sequence",
			"artificial intelligence", "neural network", "computer science",
			"algorithm", "machine learning", "programming language",
			"Roman Empire", "Renaissance", "World War II",
			"Ancient Egypt", "Industrial Revolution",
			"solar system", "continent", "ocean", "climate",
			"logic", "analogy", "metaphor", "reasoning",
			"România", "București", "Carpați", "Dunărea",
			"istoria României", "Mihai Eminescu",
		},
		AutoSeedDatasets: []string{
			"tatsu-lab/alpaca",
			"gsm8k",
			"hellaswag",
		},
		AutoSearchLangs:     []string{"en", "ro"},
		AutoHFRowsPerDS:     20,
		AutoLearnInterval:   30,
		AutoMaxGapsPerCycle: 3,

		// Web server defaults
		WebPort:     "8080",
		WebBindAddr: "127.0.0.1",

		// WebLearner defaults (previously hardcoded in web_learner.go)
		WebLearnerTimeoutSecs: 10,
		WebLearnerRateLimitMs: 2000,
		WebLearnerBodyLimitMB: 5,
		WebLearnerUserAgent:   "NexusCortex/1.0 (autonomous learner)",
		WebLearnerWikiBaseURL: "wikipedia.org",
		WebLearnerHFSearchURL: "https://huggingface.co/api/datasets",
		WebLearnerHFRowsURL:   "https://datasets-server.huggingface.co/rows",
	}
}

// Validate checks the configuration for internal consistency.
// Returns an error if any constraints are violated.
func (c Config) Validate() error {
	// Dimensional constraints
	if c.SDRSize <= 0 || c.SDRSize > 1_000_000 {
		return fmt.Errorf("SDRSize must be in [1, 1000000], got %d", c.SDRSize)
	}
	if c.ActiveCount <= 0 || c.ActiveCount > c.SDRSize {
		return fmt.Errorf("ActiveCount must be in [1, SDRSize=%d], got %d", c.SDRSize, c.ActiveCount)
	}
	if c.MaxMemories <= 0 || c.MaxMemories > 10_000_000 {
		return fmt.Errorf("MaxMemories must be in [1, 10000000], got %d", c.MaxMemories)
	}
	if c.MaxGenWords < 0 || c.MaxGenWords > 10_000 {
		return fmt.Errorf("MaxGenWords must be in [0, 10000], got %d", c.MaxGenWords)
	}

	// Network/neuron constraints
	if c.PrefrontalNetSize <= 0 || c.PrefrontalNetSize > 1_000_000 {
		return fmt.Errorf("PrefrontalNetSize must be in [1, 1000000], got %d", c.PrefrontalNetSize)
	}
	if c.NeuronMinThreshold > c.NeuronMaxThreshold {
		return fmt.Errorf("NeuronMinThreshold (%d) > NeuronMaxThreshold (%d)", c.NeuronMinThreshold, c.NeuronMaxThreshold)
	}
	if c.NeuronMinLeak > c.NeuronMaxLeak {
		return fmt.Errorf("NeuronMinLeak (%d) > NeuronMaxLeak (%d)", c.NeuronMinLeak, c.NeuronMaxLeak)
	}
	if c.SynapseMinWeight > c.SynapseMaxWeight {
		return fmt.Errorf("SynapseMinWeight (%d) > SynapseMaxWeight (%d)", c.SynapseMinWeight, c.SynapseMaxWeight)
	}
	if c.SynapseMinDelay > c.SynapseMaxDelay {
		return fmt.Errorf("SynapseMinDelay (%d) > SynapseMaxDelay (%d)", c.SynapseMinDelay, c.SynapseMaxDelay)
	}
	if c.PredictorUnionDepth > c.PredictorWindowSize {
		return fmt.Errorf("PredictorUnionDepth (%d) > PredictorWindowSize (%d)", c.PredictorUnionDepth, c.PredictorWindowSize)
	}

	// Autonomous learner constraints
	if c.AutoHFRowsPerDS < 0 || c.AutoHFRowsPerDS > 100_000 {
		return fmt.Errorf("AutoHFRowsPerDS must be in [0, 100000], got %d", c.AutoHFRowsPerDS)
	}
	if c.AutoLearnInterval < 0 {
		return fmt.Errorf("AutoLearnInterval must be >= 0, got %d", c.AutoLearnInterval)
	}
	if c.AutoMaxGapsPerCycle < 0 {
		return fmt.Errorf("AutoMaxGapsPerCycle must be >= 0, got %d", c.AutoMaxGapsPerCycle)
	}

	// Web server constraints
	if c.WebPort != "" {
		p, err := strconv.Atoi(c.WebPort)
		if err != nil || p < 1 || p > 65535 {
			return fmt.Errorf("WebPort must be a valid port [1-65535], got %q", c.WebPort)
		}
	}

	return nil
}
