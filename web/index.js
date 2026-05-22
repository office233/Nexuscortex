/* =====================================================================
   Nexus Cortex — Neural Dashboard Javascript
   ===================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Cache
    const headerFlowState = document.getElementById('header-flow-state');
    const headerClock = document.getElementById('header-clock');
    
    // Chat DOM
    const chatLog = document.getElementById('chat-log');
    const chatInput = document.getElementById('chat-input');
    const btnSend = document.getElementById('btn-send');
    const learningToggle = document.getElementById('learning-toggle');
    
    // Vitals DOM
    const vitalSynapticMass = document.getElementById('vital-synaptic-mass');
    const vitalActiveSynapses = document.getElementById('vital-active-synapses');
    const vitalMemories = document.getElementById('vital-memories');
    const memoryProgressBar = document.getElementById('memory-progress-bar');
    const vitalMemoryRatio = document.getElementById('vital-memory-ratio');
    const vitalVocabSize = document.getElementById('vital-vocab-size');
    const vitalBrocaRules = document.getElementById('vital-broca-rules');
    const vitalPrefrontalNeurons = document.getElementById('vital-prefrontal-neurons');
    const vitalPrefrontalSynapses = document.getElementById('vital-prefrontal-synapses');
    
    // Workspace DOM
    const salienceMeter = document.getElementById('salience-meter');
    const vitalSurpriseLevel = document.getElementById('vital-surprise-level');
    const vitalFocusTarget = document.getElementById('vital-focus-target');
    
    // Emotion DOM
    const vitalMood = document.getElementById('vital-mood');
    const moodPointer = document.getElementById('mood-pointer');
    const moodValenceVal = document.getElementById('mood-valence-val');
    const moodArousalVal = document.getElementById('mood-arousal-val');
    
    // Drives DOM
    const driveSleepPressure = document.getElementById('drive-sleep-pressure');
    const barSleepPressure = document.getElementById('bar-sleep-pressure');
    const driveAlertness = document.getElementById('drive-alertness');
    const barAlertness = document.getElementById('bar-alertness');
    const driveCuriosity = document.getElementById('drive-curiosity');
    const barCuriosity = document.getElementById('bar-curiosity');
    const btnSleep = document.getElementById('btn-sleep');
    
    // Sleep Overlay DOM
    const sleepOverlay = document.getElementById('sleep-overlay');
    const sleepConsole = document.getElementById('sleep-console');
    const sleepStatsDelta = document.getElementById('sleep-stats-delta');

    let pollingInterval = null;
    let isSleeping = false;
    let nexusToken = '';

    // Helper: Escapes HTML tags to prevent XSS
    function escapeHTML(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;')
                  .replace(/'/g, '&#039;');
    }

    // Helper: Formats numbers with commas (e.g. 1000 -> 1,000)
    function formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }

    // ─────────────────────────────────────────────────────────────────
    // 1. UPDATE TELEMETRY (Stats Engine)
    // ─────────────────────────────────────────────────────────────────
    function updateDashboard(stats) {
        if (isSleeping) return; // Prevent overwriting during consolidation

        // Flow state indicator
        if (stats.is_in_flow) {
            headerFlowState.innerText = "IN FLOW";
            headerFlowState.className = "value text-glow-green";
        } else {
            headerFlowState.innerText = "Active Flow";
            headerFlowState.className = "value text-glow-cyan";
        }

        // Clock tick
        headerClock.innerText = `Tick ${stats.rhythm_tick}`;

        // Left sidebar & dashboard metric outputs
        vitalSynapticMass.innerText = formatNumber(stats.total_synaptic_weight);
        vitalActiveSynapses.innerText = `${formatNumber(stats.brain_active_synapses)} active synapses`;
        
        vitalMemories.innerText = formatNumber(stats.hippocampus_memories);
        
        const memoryRatio = (stats.hippocampus_memories / stats.hippocampus_max_memories) * 100;
        memoryProgressBar.style.width = `${Math.min(memoryRatio, 100)}%`;
        vitalMemoryRatio.innerText = `${formatNumber(stats.hippocampus_memories)} / ${formatNumber(stats.hippocampus_max_memories)} stored`;

        vitalVocabSize.innerText = `${formatNumber(stats.vocab_size)} words`;
        vitalBrocaRules.innerText = `${formatNumber(stats.broca_patterns)} generation patterns`;
        
        vitalPrefrontalNeurons.innerText = `${formatNumber(stats.prefrontal_neurons)} neurons`;
        vitalPrefrontalSynapses.innerText = `${formatNumber(stats.prefrontal_synapses)} synapses`;

        // Prediction Error/Surprise Bar
        const surprisePercent = (stats.surprise_level / 255) * 100;
        salienceMeter.style.width = `${surprisePercent}%`;
        vitalSurpriseLevel.innerText = `${stats.surprise_level} / 255`;

        // Last Focus Target (Global Attention)
        vitalFocusTarget.innerText = stats.last_focus_target ? `[ ${stats.last_focus_target.toUpperCase()} ]` : "(empty)";

        // Emotional Compass Vector Mapping
        // Valence: -128 (left) to 127 (right) -> map to 5% - 95%
        const valence = stats.valence;
        const leftPercent = 50 + (valence / 128) * 45; 
        
        // Arousal: 0 (bottom) to 255 (top) -> map to 5% - 95%
        const arousal = stats.arousal;
        const bottomPercent = 5 + (arousal / 255) * 90;

        moodPointer.style.left = `${leftPercent}%`;
        moodPointer.style.bottom = `${bottomPercent}%`;

        // Update Mood labels and Pointer color based on quadrants
        vitalMood.innerText = stats.emotional_mood;
        moodValenceVal.innerText = valence;
        moodArousalVal.innerText = arousal;

        // Apply visual pointer colors
        if (valence >= 0) {
            if (arousal >= 128) {
                // Excited/Happy -> Greenish Teal
                moodPointer.style.backgroundColor = 'var(--neon-cyan)';
                moodPointer.style.boxShadow = '0 0 15px var(--neon-cyan), 0 0 30px var(--neon-cyan)';
                vitalMood.className = 'text-glow-cyan';
            } else {
                // Calm/Relaxed -> Emerald
                moodPointer.style.backgroundColor = 'var(--neon-green)';
                moodPointer.style.boxShadow = '0 0 15px var(--neon-green), 0 0 30px var(--neon-green)';
                vitalMood.className = 'text-glow-green';
            }
        } else {
            if (arousal >= 128) {
                // Angry/Excited -> Crimson Orange
                moodPointer.style.backgroundColor = 'var(--neon-red)';
                moodPointer.style.boxShadow = '0 0 15px var(--neon-red), 0 0 30px var(--neon-red)';
                vitalMood.className = 'text-glow-red';
            } else {
                // Depressed/Sad -> Royal Violet
                moodPointer.style.backgroundColor = 'var(--neon-violet)';
                moodPointer.style.boxShadow = '0 0 15px var(--neon-violet), 0 0 30px var(--neon-violet)';
                vitalMood.className = 'text-glow-purple';
            }
        }

        // Biological drives
        driveSleepPressure.innerText = `${stats.sleep_pressure}%`;
        barSleepPressure.style.width = `${stats.sleep_pressure}%`;

        driveAlertness.innerText = `${stats.alertness}%`;
        barAlertness.style.width = `${stats.alertness}%`;

        const curiosityPercent = Math.round((stats.curiosity_level / 255) * 100);
        driveCuriosity.innerText = `${curiosityPercent}%`;
        barCuriosity.style.width = `${curiosityPercent}%`;
    }

    // Standard Polling Call
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats', {
                headers: {
                    'X-Nexus-Token': nexusToken
                }
            });
            if (res.ok) {
                const stats = await res.json();
                updateDashboard(stats);
            }
        } catch (err) {
            console.error("Failed to fetch biological stats:", err);
        }
    }

    async function fetchToken() {
        try {
            const res = await fetch('/api/token');
            if (res.ok) {
                const data = await res.json();
                nexusToken = data.token;
            } else {
                console.error("Token fetch returned non-OK status:", res.status);
            }
        } catch (err) {
            console.error("Failed to fetch API security token:", err);
        }
    }

    // Start stats polling
    function startPolling() {
        if (pollingInterval) clearInterval(pollingInterval);
        fetchStats();
        pollingInterval = setInterval(fetchStats, 1000);
    }

    // ─────────────────────────────────────────────────────────────────
    // 2. CHAT & COGNITIVE INTERACTION
    // ─────────────────────────────────────────────────────────────────
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isSleeping) return;

        // Reset input immediately
        chatInput.value = '';
        btnSend.disabled = true;
        chatInput.disabled = true;

        // Append User bubble
        const userDiv = document.createElement('div');
        userDiv.className = 'message user-msg';
        userDiv.innerHTML = `<p>${escapeHTML(text)}</p>`;
        chatLog.appendChild(userDiv);
        chatLog.scrollTop = chatLog.scrollHeight;

        const isLearningMode = learningToggle.checked;

        try {
            if (isLearningMode) {
                // ── Passive Learning absorbing pipeline ─────────────────────
                const response = await fetch('/api/learn', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'X-Nexus-Token': nexusToken
                    },
                    body: JSON.stringify({ message: text })
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    // Append lightweight semantic absorption confirm bubble
                    const systemDiv = document.createElement('div');
                    systemDiv.className = 'message system-msg';
                    systemDiv.innerHTML = `<p>📝 Absorbed text segment: "${escapeHTML(text)}" successfully integrated into semantic/word chains.</p>`;
                    chatLog.appendChild(systemDiv);
                    
                    // Fast update UI state
                    if (data.stats) {
                        updateDashboard(data.stats);
                    }
                } else {
                    throw new Error("Learning backend error");
                }
            } else {
                // ── Cognitive dialogue pipeline ─────────────────────────────
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'X-Nexus-Token': nexusToken
                    },
                    body: JSON.stringify({ message: text })
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    // Append Organism response bubble
                    const orgDiv = document.createElement('div');
                    orgDiv.className = 'message organism-msg';
                    orgDiv.setAttribute('data-topic', data.stats ? data.stats.last_focus_target : '');
                    orgDiv.setAttribute('data-response', data.response);
                    orgDiv.setAttribute('data-prompt', text);
                    orgDiv.innerHTML = `
                        <p>${escapeHTML(data.response)}</p>
                        <div class="msg-telemetry">
                            <span>🎯 Conf: ${data.confidence}</span>
                            <span>⚡ Surprise: ${data.surprise}</span>
                            <span>🧬 Source: ${data.source}</span>
                        </div>
                        <div class="feedback-controls">
                            <button class="feedback-btn up-btn" title="Reinforce this response (LTP)">👍</button>
                            <button class="feedback-btn down-btn" title="Correct this response (LTD)">👎</button>
                        </div>
                    `;
                    chatLog.appendChild(orgDiv);

                    // Fast update UI state
                    if (data.stats) {
                        updateDashboard(data.stats);
                    }
                } else {
                    throw new Error("Dialogue backend error");
                }
            }
        } catch (err) {
            // Error bubble
            const errorDiv = document.createElement('div');
            errorDiv.className = 'message system-msg';
            errorDiv.style.borderColor = 'var(--neon-red)';
            errorDiv.style.color = 'var(--neon-red)';
            errorDiv.innerHTML = `<p>⚠️ Cognitive interface disconnected: ${escapeHTML(err.message)}</p>`;
            chatLog.appendChild(errorDiv);
        } finally {
            btnSend.disabled = false;
            chatInput.disabled = false;
            chatInput.focus();
            chatLog.scrollTop = chatLog.scrollHeight;
        }
    }

    // Send handlers
    btnSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });


    // ─────────────────────────────────────────────────────────────────
    // 3. SLEEP CONSOLIDATION SEQUENCE
    // ─────────────────────────────────────────────────────────────────
    async function triggerSleep() {
        if (isSleeping) return;
        isSleeping = true;
        
        // Stop stats polling during consolidated state transitions
        if (pollingInterval) clearInterval(pollingInterval);

        // Turn on fullscreen layout
        sleepOverlay.classList.add('active');
        sleepConsole.innerHTML = '';
        sleepStatsDelta.classList.remove('active');
        sleepStatsDelta.innerHTML = '';

        // Add start lines
        addConsoleLine("> Starting biological sleep cycle...");
        addConsoleLine("> Discharging prefrontal thinking fields...");
        
        try {
            // Trigger Go sleep API endpoint
            const res = await fetch('/api/sleep', { 
                method: 'POST',
                headers: { 'X-Nexus-Token': nexusToken }
            });
            if (!res.ok) throw new Error("Consolidation routine crashed");
            const delta = await res.json();
            
            // Console print speed transition runner
            const sleepProcessLogs = delta.console_logs || [
                "Replaying episodic sequences in Hippocampus...",
                "Running Long-Term Potentiation (LTP) thresholds...",
                "Generalizing episodic tracks into semantic neocortex...",
                "Pruning redundant cache directories in Cerebellum...",
                "Evicting weak prefrontal reservoir connections...",
                "Normalizing emotional valence vectors...",
                "Biological clock reset. Consolidating saved parameters..."
            ];

            let lineIndex = 0;
            const logTimer = setInterval(() => {
                if (lineIndex < sleepProcessLogs.length) {
                    addConsoleLine(`> ${sleepProcessLogs[lineIndex]}`);
                    lineIndex++;
                } else {
                    clearInterval(logTimer);
                    // Show stats delta cards
                    renderSleepDeltas(delta);
                }
            }, 300);

        } catch (err) {
            addConsoleLine(`> ⚠️ CRITICAL FAULT: ${err.message}`);
            addConsoleLine("> Aborting sleep cycle to preserve state...");
            
            // Allow clicking to escape the crash
            const wakeHint = document.createElement('div');
            wakeHint.className = 'delta-close-hint';
            wakeHint.innerText = 'Click anywhere to return to Dashboard';
            sleepConsole.appendChild(wakeHint);
            
            const abortWake = () => {
                sleepOverlay.classList.remove('active');
                isSleeping = false;
                startPolling();
                sleepOverlay.removeEventListener('click', abortWake);
            };
            sleepOverlay.addEventListener('click', abortWake);
        }
    }

    function addConsoleLine(text) {
        const line = document.createElement('div');
        line.className = 'line';
        line.innerText = text;
        sleepConsole.appendChild(line);
        sleepConsole.scrollTop = sleepConsole.scrollHeight;
    }

    function renderSleepDeltas(delta) {
        // Construct visual pre-post changes
        const memoriesPre = delta.pre.hippocampus_memories;
        const memoriesPost = delta.post.hippocampus_memories;
        const memoriesDiff = memoriesPost - memoriesPre;
        
        const synapsesPre = delta.pre.prefrontal_synapses;
        const synapsesPost = delta.post.prefrontal_synapses;
        const synapsesDiff = synapsesPost - synapsesPre;

        const memoryDiffHTML = memoriesDiff < 0 
            ? `<span class="change change-prune">${memoriesDiff} pruned</span>` 
            : `<span class="change change-stable">±0 consolidated</span>`;

        const synapsesDiffHTML = synapsesDiff < 0 
            ? `<span class="change change-prune">${synapsesDiff} pruned</span>` 
            : `<span class="change change-stable">±0 optimized</span>`;

        sleepStatsDelta.innerHTML = `
            <div class="delta-card">
                <span class="label">Episodic Pruning</span>
                <span class="value">${memoriesPre} ➔ ${memoriesPost}</span>
                ${memoryDiffHTML}
            </div>
            <div class="delta-card">
                <span class="label">Prefrontal Synapses</span>
                <span class="value">${synapsesPre} ➔ ${synapsesPost}</span>
                ${synapsesDiffHTML}
            </div>
        `;
        
        sleepStatsDelta.classList.add('active');

        // Append wake up prompt
        const hint = document.createElement('div');
        hint.className = 'delta-close-hint';
        hint.innerText = 'Consolidation complete. Click anywhere to Wake Up';
        sleepStatsDelta.appendChild(hint);

        // Click handler to wake up
        const wakeUp = () => {
            sleepOverlay.classList.remove('active');
            isSleeping = false;
            
            // Re-render dashboard stats immediately
            updateDashboard(delta.post);
            
            // Resume regular updates
            startPolling();
            sleepOverlay.removeEventListener('click', wakeUp);
        };
        
        // Wait a small moment so clicks during active animation don't accidentally close it
        setTimeout(() => {
            sleepOverlay.addEventListener('click', wakeUp);
        }, 500);
    }

    btnSleep.addEventListener('click', triggerSleep);

    // ─────────────────────────────────────────────────────────────────
    // 4. HUMAN FEEDBACK & CORRECTION MODAL (Task 10)
    // ─────────────────────────────────────────────────────────────────
    const correctionModal = document.getElementById('correction-modal');
    const correctionPromptText = document.getElementById('correction-prompt-text');
    const correctionResponseText = document.getElementById('correction-response-text');
    const correctionInput = document.getElementById('correction-input');
    const btnCorrectionCancel = document.getElementById('btn-correction-cancel');
    const btnCorrectionSubmit = document.getElementById('btn-correction-submit');
    const btnSelfTrain = document.getElementById('btn-selftrain');
    
    let activeFeedbackBtn = null;
    let activeTopic = '';
    let activeResponse = '';

    // Handle thumbs-up/down button clicks via Event Delegation inside chat-log
    chatLog.addEventListener('click', (e) => {
        const btn = e.target.closest('.feedback-btn');
        if (!btn) return;

        const msgDiv = btn.closest('.organism-msg');
        if (!msgDiv) return;

        const topic = msgDiv.getAttribute('data-topic') || '';
        const responseText = msgDiv.getAttribute('data-response') || '';
        const userPrompt = msgDiv.getAttribute('data-prompt') || '';

        if (btn.classList.contains('up-btn')) {
            // Positive feedback (thumbs up) -> dopamine reward LTP
            submitFeedback(topic, responseText, true, '');
            
            // Apply instant premium cybernetic glow styling to the clicked button
            btn.style.background = 'rgba(0, 242, 254, 0.4)';
            btn.style.borderColor = 'var(--neon-cyan)';
            btn.style.boxShadow = '0 0 10px var(--neon-cyan)';
            const sibling = btn.nextElementSibling || btn.previousElementSibling;
            if (sibling) {
                sibling.style.opacity = '0.2';
                sibling.style.pointerEvents = 'none';
            }
            btn.style.pointerEvents = 'none';
        } else if (btn.classList.contains('down-btn')) {
            // Negative feedback (thumbs down) -> LTD + correction prompt
            openCorrectionModal(topic, responseText, userPrompt, btn);
        }
    });

    function openCorrectionModal(topic, responseText, userPrompt, btn) {
        activeFeedbackBtn = btn;
        activeTopic = topic;
        activeResponse = responseText;

        correctionPromptText.innerText = `"${userPrompt}"`;
        correctionResponseText.innerText = `"${responseText}"`;
        correctionInput.value = '';

        correctionModal.classList.add('active');
        setTimeout(() => correctionInput.focus(), 100);
    }

    function closeCorrectionModal() {
        correctionModal.classList.remove('active');
        activeFeedbackBtn = null;
    }

    btnCorrectionCancel.addEventListener('click', closeCorrectionModal);

    btnCorrectionSubmit.addEventListener('click', () => {
        const correctText = correctionInput.value.trim();
        if (!correctText) return; // correction required to submit

        submitFeedback(activeTopic, activeResponse, false, correctText);
        
        // Apply instant red pruning visual styles to the clicked thumbs down button
        if (activeFeedbackBtn) {
            activeFeedbackBtn.style.background = 'rgba(255, 88, 88, 0.4)';
            activeFeedbackBtn.style.borderColor = 'var(--neon-red)';
            activeFeedbackBtn.style.boxShadow = '0 0 10px var(--neon-red)';
            const sibling = activeFeedbackBtn.previousElementSibling || activeFeedbackBtn.nextElementSibling;
            if (sibling) {
                sibling.style.opacity = '0.2';
                sibling.style.pointerEvents = 'none';
            }
            activeFeedbackBtn.style.pointerEvents = 'none';
        }

        closeCorrectionModal();
    });

    // Close modal on clicking outer overlay
    correctionModal.addEventListener('click', (e) => {
        if (e.target === correctionModal) {
            closeCorrectionModal();
        }
    });

    async function submitFeedback(topic, responseText, positive, correctText) {
        try {
            const res = await fetch('/api/feedback', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Nexus-Token': nexusToken
                },
                body: JSON.stringify({
                    topic: topic,
                    responseText: responseText,
                    positive: positive,
                    correctText: correctText
                })
            });

            if (res.ok) {
                const data = await res.json();
                
                // Append dynamic visual dopamine confirmation inside dialogue console
                const notifyDiv = document.createElement('div');
                notifyDiv.className = 'message system-msg';
                if (positive) {
                    notifyDiv.style.borderColor = 'var(--neon-green)';
                    notifyDiv.style.color = 'var(--neon-green)';
                    notifyDiv.style.background = 'rgba(56, 239, 125, 0.05)';
                    notifyDiv.innerHTML = `<p>⚡ <b>Dopaminergic LTP Sequence Reinforced</b> for topic: "${escapeHTML(topic)}" (+50 synaptic weight boost, Mood boosted)</p>`;
                } else {
                    notifyDiv.style.borderColor = 'var(--neon-red)';
                    notifyDiv.style.color = 'var(--neon-red)';
                    notifyDiv.style.background = 'rgba(255, 88, 88, 0.05)';
                    notifyDiv.innerHTML = `<p>✂️ <b>Synaptic LTD Decayed</b> (-40 weight drop). Cerebellum Cache Evicted. Passively learning correction...</p>`;
                }
                chatLog.appendChild(notifyDiv);
                chatLog.scrollTop = chatLog.scrollHeight;

                // Sync new biological parameters
                if (data.stats) {
                    updateDashboard(data.stats);
                }
            } else {
                throw new Error("Dialogue reward handler rejected packet");
            }
        } catch (err) {
            console.error("Dopamine feed transmission crashed:", err);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 5. PREFRONTAL SELF-REFLECTION TRAINING (Task 11)
    // ─────────────────────────────────────────────────────────────────
    async function triggerSelfTrain() {
        if (isSleeping) return;
        isSleeping = true;
        
        if (pollingInterval) clearInterval(pollingInterval);

        // Fetch sleep DOM components to dynamically transform into self-reflection console
        const sleepIcon = sleepOverlay.querySelector('.sleep-icon');
        const sleepTitle = sleepOverlay.querySelector('h2');
        const sleepSub = sleepOverlay.querySelector('.sleep-sub');

        // Swap to Prefrontal Reflection Mode
        sleepIcon.innerText = "🔮";
        sleepTitle.innerText = "Prefrontal Self-Reflection Active";
        sleepTitle.style.background = "linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-violet) 100%)";
        sleepTitle.style.webkitBackgroundClip = "text";
        sleepTitle.style.webkitTextFillColor = "transparent";
        sleepSub.innerText = "Simulating concept phrasings internally & testing attractor stability.";

        sleepOverlay.classList.add('active');
        sleepConsole.innerHTML = '';
        sleepStatsDelta.classList.remove('active');
        sleepStatsDelta.innerHTML = '';

        addConsoleLine("> Engaging prefrontal neural reservoir...");
        addConsoleLine("> Commencing sleep-deliberation on active cognitive tracks...");
        
        try {
            const res = await fetch('/api/selftrain', { 
                method: 'POST',
                headers: { 'X-Nexus-Token': nexusToken }
            });
            if (!res.ok) throw new Error("Autonomous reflection process faulted");
            const delta = await res.json();
            
            const reflectionLogs = delta.console_logs || [];

            let lineIndex = 0;
            const logTimer = setInterval(() => {
                if (lineIndex < reflectionLogs.length) {
                    addConsoleLine(`> ${reflectionLogs[lineIndex]}`);
                    lineIndex++;
                } else {
                    clearInterval(logTimer);
                    renderSelfTrainDeltas(delta);
                }
            }, 300);

        } catch (err) {
            addConsoleLine(`> ⚠️ CRITICAL COGNITIVE BLOCK: ${err.message}`);
            addConsoleLine("> Discharging reservoir charges. Resetting prefrontal field...");
            
            const wakeHint = document.createElement('div');
            wakeHint.className = 'delta-close-hint';
            wakeHint.innerText = 'Click anywhere to return to Dashboard';
            sleepConsole.appendChild(wakeHint);
            
            const abortWake = () => {
                sleepOverlay.classList.remove('active');
                
                // Reset elements back to standard sleep
                sleepIcon.innerText = "💤";
                sleepTitle.innerText = "Consolidating Semantic Neocortex...";
                sleepTitle.style.background = "";
                sleepTitle.style.webkitBackgroundClip = "";
                sleepTitle.style.webkitTextFillColor = "";
                sleepSub.innerText = "Episodic memories are being generalized and transferred.";
                
                isSleeping = false;
                startPolling();
                sleepOverlay.removeEventListener('click', abortWake);
            };
            sleepOverlay.addEventListener('click', abortWake);
        }
    }

    function renderSelfTrainDeltas(delta) {
        const synapsesPre = delta.pre.total_synaptic_weight;
        const synapsesPost = delta.post.total_synaptic_weight;
        const synapsesDiff = synapsesPost - synapsesPre;

        const synapseDiffHTML = synapsesDiff !== 0 
            ? `<span class="change ${synapsesDiff > 0 ? 'change-stable' : 'change-prune'}">${synapsesDiff > 0 ? '+' : ''}${formatNumber(synapsesDiff)} synapses</span>` 
            : `<span class="change change-stable">±0 stable</span>`;

        sleepStatsDelta.innerHTML = `
            <div class="delta-card">
                <span class="label">Total Synaptic Weight</span>
                <span class="value font-mono">${formatNumber(synapsesPre)} ➔ ${formatNumber(synapsesPost)}</span>
                ${synapseDiffHTML}
            </div>
            <div class="delta-card">
                <span class="label">Deliberation Outcome</span>
                <span class="value" style="font-size: 0.95rem; font-family: inherit; font-weight: 600; margin-top: 8px; text-align: left; width: 100%;">
                    Consolidated: <span style="color: var(--neon-green); font-weight: 700;">${delta.consolidated} tracks</span><br/>
                    Pruned: <span style="color: var(--neon-red); font-weight: 700;">${delta.pruned} paths</span>
                </span>
            </div>
        `;
        
        sleepStatsDelta.classList.add('active');

        const hint = document.createElement('div');
        hint.className = 'delta-close-hint';
        hint.innerText = 'Prefrontal stabilizing complete. Click anywhere to Wake Up';
        sleepStatsDelta.appendChild(hint);

        const wakeUp = () => {
            sleepOverlay.classList.remove('active');
            
            // Restore visual layout parameters
            const sleepIcon = sleepOverlay.querySelector('.sleep-icon');
            const sleepTitle = sleepOverlay.querySelector('h2');
            const sleepSub = sleepOverlay.querySelector('.sleep-sub');
            sleepIcon.innerText = "💤";
            sleepTitle.innerText = "Consolidating Semantic Neocortex...";
            sleepTitle.style.background = "";
            sleepTitle.style.webkitBackgroundClip = "";
            sleepTitle.style.webkitTextFillColor = "";
            sleepSub.innerText = "Episodic memories are being generalized and transferred.";

            isSleeping = false;
            updateDashboard(delta.post);
            startPolling();
            sleepOverlay.removeEventListener('click', wakeUp);
        };
        
        setTimeout(() => {
            sleepOverlay.addEventListener('click', wakeUp);
        }, 500);
    }

    btnSelfTrain.addEventListener('click', triggerSelfTrain);

    // Initial setup
    async function initialize() {
        await fetchToken();
        startPolling();
    }
    initialize();
});
