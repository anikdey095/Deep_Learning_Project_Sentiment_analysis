(() => {
  "use strict";

  const EMOTIONS = {
    sadness: { label: "Sadness", emoji: "😢", color: "#3b82f6" },
    joy: { label: "Joy", emoji: "😄", color: "#f59e0b" },
    love: { label: "Love", emoji: "❤️", color: "#f43f5e" },
    anger: { label: "Anger", emoji: "😠", color: "#ef4444" },
    fear: { label: "Fear", emoji: "😨", color: "#a855f7" },
    surprise: { label: "Surprise", emoji: "😲", color: "#06b6d4" },
  };

  // DOM Elements
  const el = {
    statusDot: document.getElementById("statusDot"),
    modelStatusPill: document.getElementById("modelStatusPill"),
    soundToggleBtn: document.getElementById("soundToggleBtn"),
    soundIcon: document.getElementById("soundIcon"),
    soundLabel: document.getElementById("soundLabel"),
    presetChips: document.getElementById("presetChips"),
    neuralOrb: document.getElementById("neuralOrb"),
    orbSymbol: document.getElementById("orbSymbol"),
    orbStatusText: document.getElementById("orbStatusText"),
    textInput: document.getElementById("textInput"),
    clearBtn: document.getElementById("clearBtn"),
    charCount: document.getElementById("charCount"),
    wordCount: document.getElementById("wordCount"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    errorBanner: document.getElementById("errorBanner"),
    errorText: document.getElementById("errorText"),
    resultSection: document.getElementById("resultSection"),
    latencyBadge: document.getElementById("latencyBadge"),
    emotionAvatar: document.getElementById("emotionAvatar"),
    emotionName: document.getElementById("emotionName"),
    confidenceBadge: document.getElementById("confidenceBadge"),
    confidenceMeterFill: document.getElementById("confidenceMeterFill"),
    echoedText: document.getElementById("echoedText"),
    copyBtn: document.getElementById("copyBtn"),
    copyIcon: document.getElementById("copyIcon"),
    copyText: document.getElementById("copyText"),
    newAnalysisBtn: document.getElementById("newAnalysisBtn"),
    spectrumBars: document.getElementById("spectrumBars"),
    historySection: document.getElementById("historySection"),
    historyList: document.getElementById("historyList"),
    clearHistoryBtn: document.getElementById("clearHistoryBtn"),
  };

  // Safe localStorage helper for Edge InPrivate / Tracking Prevention modes
  function safeGetItem(key, fallback) {
    try {
      const v = window.localStorage ? window.localStorage.getItem(key) : null;
      return v !== null ? v : fallback;
    } catch (e) {
      return fallback;
    }
  }

  function safeSetItem(key, value) {
    try {
      if (window.localStorage) {
        window.localStorage.setItem(key, value);
      }
    } catch (e) {}
  }

  let modelReady = false;
  let audioEnabled = safeGetItem("moodline_audio", "true") !== "false";
  let historyItems = [];
  let currentResult = null;

  // Setup Audio Synthesizer via Web Audio API
  let audioCtx = null;
  function playTone(freq1 = 523.25, freq2 = 659.25) {
    if (!audioEnabled) return;
    try {
      if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;
        audioCtx = new AudioContextClass();
      }
      if (audioCtx.state === "suspended") {
        audioCtx.resume();
      }
      const now = audioCtx.currentTime;

      // Master gain
      const masterGain = audioCtx.createGain();
      masterGain.gain.setValueAtTime(0.08, now);
      masterGain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
      masterGain.connect(audioCtx.destination);

      // Osc 1
      const osc1 = audioCtx.createOscillator();
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(freq1, now);
      osc1.connect(masterGain);
      osc1.start(now);
      osc1.stop(now + 0.5);

      // Osc 2 (chord harmony)
      const osc2 = audioCtx.createOscillator();
      osc2.type = "triangle";
      osc2.frequency.setValueAtTime(freq2, now + 0.05);
      osc2.connect(masterGain);
      osc2.start(now + 0.05);
      osc2.stop(now + 0.5);
    } catch (e) {
      // Audio context silently handled if user hasn't interacted
    }
  }

  function updateAudioUI() {
    if (el.soundIcon) el.soundIcon.textContent = audioEnabled ? "🔔" : "🔕";
    if (el.soundLabel) el.soundLabel.textContent = audioEnabled ? "Audio On" : "Muted";
  }

  if (el.soundToggleBtn) {
    el.soundToggleBtn.addEventListener("click", () => {
      audioEnabled = !audioEnabled;
      safeSetItem("moodline_audio", audioEnabled ? "true" : "false");
      updateAudioUI();
      if (audioEnabled) playTone(600, 800);
    });
  }

  /* --------------------------------------------------------------------------
     1. Health Check Polling
     -------------------------------------------------------------------------- */
  async function pollHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("Health endpoint returned " + res.status);
      const data = await res.json();

      modelReady = !!data.model_loaded;
      if (modelReady) {
        setServerState("live", `BiGRU Neural Net Ready (${data.vocab_size.toLocaleString()} words in vocab)`);
      } else {
        setServerState("warming", "Warming up deep learning weights…");
        setTimeout(pollHealth, 2500);
      }
    } catch (err) {
      // Note: Edge or adblockers may block background polling, but predict endpoint still works
      modelReady = false;
      setServerState("offline", "Server connecting / waking up…");
      setTimeout(pollHealth, 5000);
    }
    syncButton();
  }

  function setServerState(status, text) {
    if (el.statusDot) el.statusDot.className = "status-dot " + (status === "live" ? "" : status);
    if (el.modelStatusPill) el.modelStatusPill.textContent = text;
    if (el.orbStatusText) {
      if (status === "live") {
        el.orbStatusText.textContent = "Ready for input";
      } else if (status === "warming") {
        el.orbStatusText.textContent = "Model warming up…";
      } else {
        el.orbStatusText.textContent = "Connecting to server…";
      }
    }
  }

  /* --------------------------------------------------------------------------
     2. Input Telemetry & Handling
     -------------------------------------------------------------------------- */
  function updateInputTelemetry() {
    const val = el.textInput ? el.textInput.value : "";
    const charLen = val.length;
    const words = val.trim().split(/\s+/).filter(Boolean).length;

    if (el.charCount) el.charCount.textContent = charLen.toLocaleString();
    if (el.wordCount) el.wordCount.textContent = words.toLocaleString();
    if (el.clearBtn) el.clearBtn.style.display = charLen > 0 ? "flex" : "none";

    syncButton();
  }

  function syncButton() {
    const val = el.textInput ? el.textInput.value.trim() : "";
    const hasText = val.length > 0;
    // Allow clicking as long as text exists (predict request will auto-wake/verify model)
    if (el.analyzeBtn) {
      el.analyzeBtn.disabled = !hasText;
    }
  }

  if (el.textInput) {
    el.textInput.addEventListener("input", updateInputTelemetry);

    el.textInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        executePrediction();
      } else if (e.key === "Escape") {
        clearInput();
      }
    });
  }
      e.preventDefault();
      executePrediction();
    } else if (e.key === "Escape") {
      clearInput();
    }
  });

  function clearInput() {
    el.textInput.value = "";
    updateInputTelemetry();
    el.textInput.focus();
    hideError();
  }

  el.clearBtn.addEventListener("click", clearInput);
  el.analyzeBtn.addEventListener("click", executePrediction);

  /* --------------------------------------------------------------------------
     3. Preset Prompt Chips
     -------------------------------------------------------------------------- */
  el.presetChips.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    const sampleText = chip.getAttribute("data-text");
    if (sampleText) {
      el.textInput.value = sampleText;
      updateInputTelemetry();
      executePrediction();
    }
  });

  /* --------------------------------------------------------------------------
     4. Prediction Pipeline
     -------------------------------------------------------------------------- */
  async function executePrediction() {
    const rawText = el.textInput ? el.textInput.value.trim() : "";
    if (!rawText) return;

    hideError();
    startThinking();

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: rawText }),
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        const message = errJson.detail || `Server returned error (${response.status})`;
        throw new Error(message);
      }

      const data = await response.json();
      modelReady = true;
      setServerState("live", "BiGRU Neural Net Ready");
      currentResult = data;
      renderPrediction(data, rawText);
      addToHistory(data, rawText);
      playTone(523.25, 783.99); // Joyful harmonic chime
    } catch (err) {
      showError(err.message || "Unable to classify sentence emotion. Please retry.");
    } finally {
      stopThinking();
    }
  }

  function startThinking() {
    el.analyzeBtn.classList.add("loading");
    el.analyzeBtn.disabled = true;
    el.analyzeBtn.querySelector(".btn-text").textContent = "Analyzing…";
    el.neuralOrb.classList.add("thinking");
    el.orbSymbol.textContent = "✦";
    el.orbStatusText.textContent = "Neural inference…";
  }

  function stopThinking() {
    el.analyzeBtn.classList.remove("loading");
    el.analyzeBtn.querySelector(".btn-text").textContent = "Read Emotion";
    el.neuralOrb.classList.remove("thinking");
    syncButton();
  }

  function renderPrediction(data, originalText) {
    const emotionKey = data.predicted_emotion;
    const meta = EMOTIONS[emotionKey] || { label: emotionKey, emoji: "✨", color: "#6366f1" };

    // Update body data-emotion attribute for dynamic ambient aura
    document.body.setAttribute("data-emotion", emotionKey);

    // Update Neural Orb
    el.orbSymbol.textContent = meta.emoji;
    el.orbStatusText.textContent = `${meta.label} (${(data.confidence * 100).toFixed(1)}%)`;

    // Update Primary Result Card
    el.emotionAvatar.textContent = meta.emoji;
    el.emotionName.textContent = meta.label;
    el.confidenceBadge.textContent = `${(data.confidence * 100).toFixed(1)}% certainty`;
    el.echoedText.textContent = originalText;
    el.latencyBadge.textContent = `⚡ ${data.latency_ms} ms`;

    // Animate certainty meter
    el.confidenceMeterFill.style.width = `${(data.confidence * 100).toFixed(1)}%`;

    // Render Probability Spectrum
    renderSpectrum(data.all_probabilities || data.all_probabilites);

    // Show result section with smooth entrance
    el.resultSection.hidden = false;
    el.resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderSpectrum(probs) {
    if (!probs) return;
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    el.spectrumBars.innerHTML = "";

    sorted.forEach(([label, prob], index) => {
      const meta = EMOTIONS[label] || { label, emoji: "✨" };
      const pct = (prob * 100).toFixed(1);

      const row = document.createElement("div");
      row.className = "spectrum-row";
      row.innerHTML = `
        <span class="spectrum-label">
          <span>${meta.emoji}</span>
          <span>${meta.label}</span>
        </span>
        <div class="spectrum-track" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
          <div class="spectrum-fill fill-${label}" style="width: 0%"></div>
        </div>
        <span class="spectrum-pct">${pct}%</span>
      `;

      el.spectrumBars.appendChild(row);

      // Staggered fill animation
      setTimeout(() => {
        const fill = row.querySelector(".spectrum-fill");
        if (fill) fill.style.width = `${pct}%`;
      }, 50 + index * 60);
    });
  }

  /* --------------------------------------------------------------------------
     5. History Drawer
     -------------------------------------------------------------------------- */
  function addToHistory(data, text) {
    historyItems.unshift({ data, text });
    if (historyItems.length > 8) historyItems.pop();
    renderHistory();
  }

  function renderHistory() {
    if (historyItems.length === 0) {
      el.historySection.hidden = true;
      return;
    }

    el.historySection.hidden = false;
    el.historyList.innerHTML = "";

    historyItems.forEach((item, idx) => {
      const meta = EMOTIONS[item.data.predicted_emotion] || { emoji: "✨" };
      const pill = document.createElement("button");
      pill.className = "history-item";
      pill.title = item.text;
      pill.innerHTML = `
        <span>${meta.emoji}</span>
        <span class="history-snippet">${escapeHtml(item.text)}</span>
      `;
      pill.addEventListener("click", () => {
        el.textInput.value = item.text;
        updateInputTelemetry();
        renderPrediction(item.data, item.text);
      });
      el.historyList.appendChild(pill);
    });
  }

  el.clearHistoryBtn.addEventListener("click", () => {
    historyItems = [];
    renderHistory();
  });

  /* --------------------------------------------------------------------------
     6. Action Utilities (Copy & Reset)
     -------------------------------------------------------------------------- */
  el.copyBtn.addEventListener("click", async () => {
    if (!currentResult) return;
    const textToCopy = `Moodline Emotion: ${currentResult.predicted_emotion.toUpperCase()} (${(currentResult.confidence * 100).toFixed(1)}% certainty)\nSentence: "${currentResult.text}"`;
    try {
      await navigator.clipboard.writeText(textToCopy);
      el.copyText.textContent = "Copied! ✓";
      setTimeout(() => {
        el.copyText.textContent = "Copy Result";
      }, 2000);
    } catch (e) {
      el.copyText.textContent = "Copy failed";
      setTimeout(() => {
        el.copyText.textContent = "Copy Result";
      }, 2000);
    }
  });

  el.newAnalysisBtn.addEventListener("click", () => {
    clearInput();
    el.textInput.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  /* --------------------------------------------------------------------------
     7. Error Handling & Helpers
     -------------------------------------------------------------------------- */
  function showError(msg) {
    el.errorText.textContent = msg;
    el.errorBanner.hidden = false;
    el.errorBanner.style.display = "flex";
  }

  function hideError() {
    el.errorBanner.hidden = true;
    el.errorBanner.style.display = "none";
    el.errorText.textContent = "";
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /* --------------------------------------------------------------------------
     Initial Boot
     -------------------------------------------------------------------------- */
  hideError();
  updateAudioUI();
  updateInputTelemetry();
  pollHealth();

})();