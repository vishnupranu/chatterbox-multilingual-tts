/**
 * app.js
 * Frontend controller for Chatterbox Z.ai Multilingual Platform.
 * Manages Chat, Code workspace, autonomous Workers, live mic recording, and voice library.
 */

// Application State
const state = {
  currentTab: 'chat',
  selectedLanguage: 'en',
  referenceAudioUrl: null,
  referenceAudioFilename: null,
  settings: {
    exaggeration: 0.5,
    cfg_weight: 0.5,
    temperature: 0.8,
    seed: 0,
    model_version: 'v3',
  },
  languages: {},
  samples: {},
  presets: [],
  chatMessages: [],
  workersInterval: null,
  mediaRecorder: null,
  audioChunks: [],
  isRecording: false,
};

// Code Workspace Templates
const CODE_TEMPLATES = {
  python: {
    title: "Python SDK (ChatterboxMultilingualTTS)",
    language: "python",
    code: `import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# 1. Initialize Multilingual Model on Apple Silicon MPS or CUDA
model = ChatterboxMultilingualTTS.from_pretrained(device="mps", t3_model="v3")

# 2. Synthesize multilingual speech across 23 languages
text = "Bonjour, ceci est le modèle multilingue de haute fidélité Chatterbox."
wav = model.generate(
    text=text,
    language_id="fr",
    exaggeration=0.5,
    cfg_weight=0.5
)

# 3. Save high-fidelity 24kHz WAV
ta.save("output_french.wav", wav, model.sr)
print("Speech rendered successfully!")`
  },
  curl: {
    title: "cURL REST API (/api/chat)",
    language: "bash",
    code: `curl -X POST "http://localhost:8080/api/chat" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "Welcome to Chatterbox Z.ai voice synthesis.",
       "language": "en",
       "exaggeration": 0.5,
       "cfg_weight": 0.5,
       "temperature": 0.8,
       "model_version": "v3"
     }' \\
     --output speech.json`
  },
  javascript: {
    title: "JavaScript / Web Client (Fetch API)",
    language: "javascript",
    code: `async function generateSpeech(text, lang = "es") {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: text,
      language: lang,
      exaggeration: 0.5,
      cfg_weight: 0.5
    })
  });

  const data = await response.json();
  console.log("Audio URL:", data.audio_url);
  const audio = new Audio(data.audio_url);
  audio.play();
}

generateSpeech("Hola mundo, el sintetizador de voz está activo!");`
  },
  cli: {
    title: "Command-Line Terminal (cli.py)",
    language: "bash",
    code: `# List all 23 supported languages
./run.sh languages

# Synthesize English
./run.sh cli -t "Hello world from terminal" -l en -o speech.wav

# Cross-lingual voice clone with reference clip
./run.sh cli -t "Guten Tag, wie geht es Ihnen?" -l de --ref-audio sample.wav -o cloned_german.wav`
  }
};

// DOM Initialization
document.addEventListener('DOMContentLoaded', async () => {
  initTabs();
  initSettingsModal();
  initPresetsModal();
  initBillingModal();
  initVoiceUpload();
  initMicRecording();
  initCodeWorkspace();
  initWorkers();
  await loadSystemStatusAndLanguages();
  await loadPresetVoices();
  await loadBillingPlansAndBalance();
  addWelcomeChatMessage();
});

// Load System Status & Languages
async function loadSystemStatusAndLanguages() {
  try {
    const [statusRes, langRes] = await Promise.all([
      fetch('/api/status'),
      fetch('/api/languages')
    ]);

    const status = await statusRes.json();
    const langData = await langRes.json();

    state.languages = langData.languages || {};
    state.samples = langData.samples || {};

    // Update status badge
    const badge = document.getElementById('system-status-badge');
    if (badge) {
      badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 inline-block mr-1.5 animate-pulse"></span> ${status.model_version.toUpperCase()} • ${status.device.toUpperCase()}`;
    }

    // Populate language dropdowns
    populateLanguageSelectors();
  } catch (err) {
    console.error("Failed to load languages/status:", err);
  }
}

// Load Preset Voices
async function loadPresetVoices() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    state.presets = data.presets || [];

    // Render sidebar presets
    const sidebarList = document.getElementById('sidebar-voice-presets-list');
    if (sidebarList) {
      sidebarList.innerHTML = state.presets.slice(0, 5).map(p => `
        <div onclick="selectPresetVoice('${p.id}')" class="px-2.5 py-1.5 rounded-lg hover:bg-white/5 cursor-pointer truncate flex items-center justify-between group transition">
          <span class="truncate">${p.avatar} ${p.name}</span>
          <span class="text-[10px] text-zinc-500 font-mono group-hover:text-indigo-400">${p.lang.toUpperCase()}</span>
        </div>
      `).join('');
    }

    // Render presets modal
    const modalContainer = document.getElementById('presets-cards-container');
    if (modalContainer) {
      modalContainer.innerHTML = state.presets.map(p => `
        <div onclick="selectPresetVoice('${p.id}')" class="p-3.5 rounded-xl glass-panel border border-white/5 hover:border-indigo-500/40 cursor-pointer transition flex flex-col justify-between group">
          <div class="flex items-start justify-between gap-2 mb-2">
            <div class="flex items-center gap-2">
              <span class="text-xl p-1.5 rounded-lg bg-white/5">${p.avatar}</span>
              <div>
                <div class="text-xs font-bold text-white group-hover:text-indigo-300 transition">${p.name}</div>
                <div class="text-[10px] text-zinc-400">${p.desc}</div>
              </div>
            </div>
            <span class="text-[10px] px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono">${p.lang.toUpperCase()}</span>
          </div>
          <div class="flex items-center justify-between mt-2 pt-2 border-t border-white/5 text-[11px] text-zinc-400">
            <span>Click to Activate Voice</span>
            <button class="px-2 py-0.5 rounded bg-indigo-600/30 text-indigo-300 group-hover:bg-indigo-600 group-hover:text-white transition">Select</button>
          </div>
        </div>
      `).join('');
    }
  } catch (err) {
    console.error("Failed to load preset voices:", err);
  }
}

function selectPresetVoice(presetId) {
  const p = state.presets.find(x => x.id === presetId);
  if (!p) return;

  state.referenceAudioUrl = p.audio;
  state.referenceAudioFilename = p.name;
  state.selectedLanguage = p.lang;

  // Update language dropdown
  const langSelect = document.getElementById('chat-language-select');
  if (langSelect) langSelect.value = p.lang;

  // Update voice badge
  const badge = document.getElementById('voice-clone-badge');
  if (badge) {
    badge.classList.remove('hidden');
    badge.classList.add('flex');
    document.getElementById('voice-clone-filename').textContent = p.name;
  }

  // Close modal
  const modal = document.getElementById('presets-modal');
  if (modal) modal.classList.add('hidden');
}

function populateLanguageSelectors() {
  const langSelect = document.getElementById('chat-language-select');
  const workerLangSelect = document.getElementById('batch-language-select');

  const optionsHtml = Object.entries(state.languages)
    .sort((a, b) => a[1].localeCompare(b[1]))
    .map(([code, name]) => `<option value="${code}" ${code === state.selectedLanguage ? 'selected' : ''}>${name} (${code})</option>`)
    .join('');

  if (langSelect) langSelect.innerHTML = optionsHtml;
  if (workerLangSelect) workerLangSelect.innerHTML = optionsHtml;

  if (langSelect) {
    langSelect.addEventListener('change', (e) => {
      state.selectedLanguage = e.target.value;
      updatePromptPlaceholder();
    });
  }
}

function updatePromptPlaceholder() {
  const input = document.getElementById('chat-input-text');
  const sample = state.samples[state.selectedLanguage];
  if (sample && input && !input.value.trim()) {
    input.placeholder = `e.g., "${sample.text}"`;
  }
}

// Tab Navigation
function initTabs() {
  const tabs = ['chat', 'code', 'workers'];
  tabs.forEach(tabName => {
    const btn = document.getElementById(`nav-btn-${tabName}`);
    if (btn) {
      btn.addEventListener('click', () => switchTab(tabName));
    }
  });
}

function switchTab(targetTab) {
  state.currentTab = targetTab;
  const tabs = ['chat', 'code', 'workers'];

  tabs.forEach(tab => {
    const btn = document.getElementById(`nav-btn-${tab}`);
    const view = document.getElementById(`workspace-view-${tab}`);
    if (btn) {
      if (tab === targetTab) {
        btn.classList.add('nav-tab-active');
        btn.classList.remove('text-zinc-400');
        btn.classList.add('text-white');
      } else {
        btn.classList.remove('nav-tab-active');
        btn.classList.remove('text-white');
        btn.classList.add('text-zinc-400');
      }
    }
    if (view) {
      if (tab === targetTab) {
        view.classList.remove('hidden');
        view.classList.add('flex');
      } else {
        view.classList.add('hidden');
        view.classList.remove('flex');
      }
    }
  });

  // Start polling workers if on workers tab
  if (targetTab === 'workers') {
    refreshWorkersJobs();
    if (!state.workersInterval) {
      state.workersInterval = setInterval(refreshWorkersJobs, 3000);
    }
  } else if (state.workersInterval) {
    clearInterval(state.workersInterval);
    state.workersInterval = null;
  }
}

// Chat Workspace logic
function addWelcomeChatMessage() {
  addChatMessage({
    role: 'assistant',
    text: "Hello! I am your Chatterbox Multilingual voice assistant. Type any text in up to 23 languages, click 🎙️ to record your voice, or select a Voice Library preset for instant zero-shot voice cloning.",
    language: 'en',
    audio_url: null,
  });
}

function addChatMessage(msg) {
  state.chatMessages.push(msg);
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  const isUser = msg.role === 'user';
  const bubble = document.createElement('div');
  bubble.className = `flex gap-3.5 my-4 w-full ${isUser ? 'justify-end' : 'justify-start'}`;

  const langBadge = msg.language ? `<span class="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-mono">${msg.language.toUpperCase()}</span>` : '';
  const durationText = msg.duration ? `<span class="text-xs text-zinc-400 font-mono">${msg.duration}s</span>` : '';

  let audioPlayerHtml = '';
  if (msg.audio_url) {
    const audioId = `audio-${Date.now()}-${Math.floor(Math.random()*1000)}`;
    audioPlayerHtml = `
      <div class="mt-3 p-3 rounded-xl bg-zinc-950/60 border border-white/5 flex flex-col gap-2">
        <div class="flex items-center justify-between gap-3">
          <div class="flex items-center gap-2">
            <button onclick="toggleAudioPlayback('${audioId}')" id="btn-play-${audioId}" class="w-9 h-9 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center transition">
              <svg id="icon-play-${audioId}" class="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
              <svg id="icon-pause-${audioId}" class="w-4 h-4 hidden" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
            </button>
            <div id="waveform-${audioId}" class="flex items-center gap-0.5 h-6">
              <span class="wave-bar"></span><span class="wave-bar"></span><span class="wave-bar"></span><span class="wave-bar"></span><span class="wave-bar"></span>
            </div>
          </div>
          <div class="flex items-center gap-2">
            <!-- Speed controls -->
            <div class="flex items-center bg-zinc-900 rounded-lg p-0.5 text-[10px] font-mono text-zinc-400">
              <button onclick="setAudioSpeed('${audioId}', 1.0)" class="px-1.5 py-0.5 rounded hover:bg-white/10">1x</button>
              <button onclick="setAudioSpeed('${audioId}', 1.25)" class="px-1.5 py-0.5 rounded hover:bg-white/10">1.25x</button>
              <button onclick="setAudioSpeed('${audioId}', 1.5)" class="px-1.5 py-0.5 rounded hover:bg-white/10">1.5x</button>
            </div>
            <span id="time-${audioId}" class="text-xs font-mono text-zinc-400">0:00</span>
            <a href="${msg.audio_url}" download="chatterbox_${msg.language || 'audio'}.wav" class="p-1.5 text-zinc-400 hover:text-white rounded-lg hover:bg-white/5 transition" title="Download WAV">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
            </a>
          </div>
        </div>
        <audio id="${audioId}" src="${msg.audio_url}" preload="auto"></audio>
      </div>
    `;
  }

  bubble.innerHTML = `
    ${!isUser ? `
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-cyan-500 flex items-center justify-center text-white font-bold text-xs shadow-lg shadow-indigo-500/20 flex-shrink-0">
        Z
      </div>
    ` : ''}
    <div class="max-w-[78%] rounded-2xl p-4 ${isUser ? 'chat-bubble-user text-white' : 'chat-bubble-ai text-zinc-200'}">
      <div class="flex items-center justify-between gap-3 mb-1">
        <span class="text-xs font-medium text-zinc-400">${isUser ? 'You' : 'Chatterbox Assistant'}</span>
        <div class="flex items-center gap-1.5">${langBadge} ${durationText}</div>
      </div>
      <p class="text-sm leading-relaxed whitespace-pre-wrap">${escapeHtml(msg.text)}</p>
      ${audioPlayerHtml}
    </div>
    ${isUser ? `
      <div class="w-8 h-8 rounded-xl bg-zinc-800 border border-white/10 flex items-center justify-center text-zinc-300 font-bold text-xs flex-shrink-0">
        U
      </div>
    ` : ''}
  `;

  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;

  // If audio is present, setup events and auto-play
  if (msg.audio_url) {
    const audios = container.querySelectorAll('audio');
    const latestAudio = audios[audios.length - 1];
    if (latestAudio) {
      setupAudioListeners(latestAudio.id);
      if (!isUser) {
        setTimeout(() => toggleAudioPlayback(latestAudio.id), 200);
      }
    }
  }
}

// Audio Speed
function setAudioSpeed(audioId, rate) {
  const audio = document.getElementById(audioId);
  if (audio) {
    audio.playbackRate = rate;
  }
}

// Audio Player Handlers
function setupAudioListeners(audioId) {
  const audio = document.getElementById(audioId);
  const timeDisplay = document.getElementById(`time-${audioId}`);
  const waveform = document.getElementById(`waveform-${audioId}`);
  const playIcon = document.getElementById(`icon-play-${audioId}`);
  const pauseIcon = document.getElementById(`icon-pause-${audioId}`);

  if (!audio) return;

  audio.addEventListener('timeupdate', () => {
    const cur = Math.floor(audio.currentTime);
    const mins = Math.floor(cur / 60);
    const secs = cur % 60;
    if (timeDisplay) timeDisplay.textContent = `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  });

  audio.addEventListener('ended', () => {
    if (waveform) waveform.classList.remove('wave-playing');
    if (playIcon) playIcon.classList.remove('hidden');
    if (pauseIcon) pauseIcon.classList.add('hidden');
  });
}

function toggleAudioPlayback(audioId) {
  const audio = document.getElementById(audioId);
  const waveform = document.getElementById(`waveform-${audioId}`);
  const playIcon = document.getElementById(`icon-play-${audioId}`);
  const pauseIcon = document.getElementById(`icon-pause-${audioId}`);
  if (!audio) return;

  if (audio.paused) {
    // Pause all other audio instances
    document.querySelectorAll('audio').forEach(a => {
      if (a.id !== audioId) a.pause();
    });
    document.querySelectorAll('[id^="waveform-"]').forEach(w => w.classList.remove('wave-playing'));
    document.querySelectorAll('[id^="icon-play-"]').forEach(p => p.classList.remove('hidden'));
    document.querySelectorAll('[id^="icon-pause-"]').forEach(p => p.classList.add('hidden'));

    audio.play();
    if (waveform) waveform.classList.add('wave-playing');
    if (playIcon) playIcon.classList.add('hidden');
    if (pauseIcon) pauseIcon.classList.remove('hidden');
  } else {
    audio.pause();
    if (waveform) waveform.classList.remove('wave-playing');
    if (playIcon) playIcon.classList.remove('hidden');
    if (pauseIcon) pauseIcon.classList.add('hidden');
  }
}

// Send Message Submission
async function handleChatSubmit() {
  const input = document.getElementById('chat-input-text');
  const btn = document.getElementById('btn-send-chat');
  if (!input || !input.value.trim()) return;

  const text = input.value.trim();
  const lang = state.selectedLanguage;
  input.value = '';

  // Render User Message
  addChatMessage({
    role: 'user',
    text: text,
    language: lang,
    audio_url: null,
  });

  // Render Loading Assistant Indicator
  const loadingIndicatorId = `loading-${Date.now()}`;
  const container = document.getElementById('chat-messages-container');
  const loadingDiv = document.createElement('div');
  loadingDiv.id = loadingIndicatorId;
  loadingDiv.className = 'flex gap-3.5 my-4 w-full justify-start items-center';
  loadingDiv.innerHTML = `
    <div class="w-8 h-8 rounded-xl bg-indigo-600/30 flex items-center justify-center text-indigo-400 font-bold text-xs flex-shrink-0 animate-pulse">
      Z
    </div>
    <div class="glass-panel rounded-2xl px-4 py-3 text-xs text-zinc-400 flex items-center gap-2">
      <svg class="animate-spin w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
      Synthesizing speech on ${state.settings.model_version.toUpperCase()}...
    </div>
  `;
  container.appendChild(loadingDiv);
  container.scrollTop = container.scrollHeight;

  if (btn) btn.disabled = true;

  try {
    const payload = {
      text: text,
      language: lang,
      audio_prompt_url: state.referenceAudioUrl,
      exaggeration: state.settings.exaggeration,
      cfg_weight: state.settings.cfg_weight,
      temperature: state.settings.temperature,
      seed: state.settings.seed,
      model_version: state.settings.model_version,
    };

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    loadingDiv.remove();

    if (!res.ok) {
      throw new Error(data.detail || "Synthesis request failed");
    }

    addChatMessage({
      role: 'assistant',
      text: data.text,
      language: data.language,
      audio_url: data.audio_url,
      duration: data.duration,
    });
  } catch (err) {
    loadingDiv.remove();
    addChatMessage({
      role: 'assistant',
      text: `⚠️ Error synthesizing audio: ${err.message}`,
      language: lang,
      audio_url: null,
    });
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Live Microphone Recording for Voice Cloning
function initMicRecording() {
  const micBtn = document.getElementById('btn-mic-record');
  const micIconIdle = document.getElementById('icon-mic-idle');
  const micIconRec = document.getElementById('icon-mic-recording');
  const badge = document.getElementById('voice-clone-badge');

  if (!micBtn) return;

  micBtn.addEventListener('click', async () => {
    if (!state.isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.audioChunks = [];
        state.mediaRecorder = new MediaRecorder(stream);

        state.mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) state.audioChunks.push(e.data);
        };

        state.mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(state.audioChunks, { type: 'audio/wav' });
          const file = new File([audioBlob], `mic_recording_${Date.now()}.wav`, { type: 'audio/wav' });
          
          const formData = new FormData();
          formData.append('file', file);

          try {
            const res = await fetch('/api/upload-voice', {
              method: 'POST',
              body: formData,
            });
            const data = await res.json();
            if (data.success) {
              state.referenceAudioUrl = data.url;
              state.referenceAudioFilename = "Mic Recording";

              if (badge) {
                badge.classList.remove('hidden');
                badge.classList.add('flex');
                document.getElementById('voice-clone-filename').textContent = "Mic Recording";
              }
            }
          } catch (err) {
            alert('Failed to upload microphone recording: ' + err.message);
          }
        };

        state.mediaRecorder.start();
        state.isRecording = true;
        micIconIdle.classList.add('hidden');
        micIconRec.classList.remove('hidden');
      } catch (err) {
        alert('Microphone access denied or unsupported: ' + err.message);
      }
    } else {
      if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
        state.mediaRecorder.stream.getTracks().forEach(t => t.stop());
      }
      state.isRecording = false;
      micIconIdle.classList.remove('hidden');
      micIconRec.classList.add('hidden');
    }
  });
}

// Voice Cloning Reference Audio Upload
function initVoiceUpload() {
  const fileInput = document.getElementById('voice-upload-input');
  const triggerBtn = document.getElementById('btn-voice-clone-trigger');
  const badge = document.getElementById('voice-clone-badge');
  const clearBtn = document.getElementById('btn-clear-voice-clone');

  if (triggerBtn && fileInput) {
    triggerBtn.addEventListener('click', () => fileInput.click());
  }

  if (fileInput) {
    fileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      try {
        triggerBtn.classList.add('opacity-50');
        const res = await fetch('/api/upload-voice', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        if (data.success) {
          state.referenceAudioUrl = data.url;
          state.referenceAudioFilename = data.filename;

          if (badge) {
            badge.classList.remove('hidden');
            badge.classList.add('flex');
            document.getElementById('voice-clone-filename').textContent = data.filename;
          }
        }
      } catch (err) {
        alert('Failed to upload reference voice: ' + err.message);
      } finally {
        triggerBtn.classList.remove('opacity-50');
      }
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      state.referenceAudioUrl = null;
      state.referenceAudioFilename = null;
      if (badge) badge.classList.add('hidden');
      if (fileInput) fileInput.value = '';
    });
  }
}

// Presets Modal
function initPresetsModal() {
  const modal = document.getElementById('presets-modal');
  const openBtn = document.getElementById('btn-open-presets');
  const closeBtn = document.getElementById('btn-close-presets');

  if (openBtn && modal) {
    openBtn.addEventListener('click', () => modal.classList.remove('hidden'));
  }
  if (closeBtn && modal) {
    closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
  }
}

// Settings Modal
function initSettingsModal() {
  const modal = document.getElementById('settings-modal');
  const openBtn = document.getElementById('btn-open-settings');
  const closeBtn = document.getElementById('btn-close-settings');
  const saveBtn = document.getElementById('btn-save-settings');

  if (openBtn && modal) {
    openBtn.addEventListener('click', () => modal.classList.remove('hidden'));
  }
  if (closeBtn && modal) {
    closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
  }
  if (saveBtn && modal) {
    saveBtn.addEventListener('click', () => {
      state.settings.exaggeration = parseFloat(document.getElementById('setting-exaggeration').value);
      state.settings.cfg_weight = parseFloat(document.getElementById('setting-cfg').value);
      state.settings.temperature = parseFloat(document.getElementById('setting-temperature').value);
      state.settings.seed = parseInt(document.getElementById('setting-seed').value) || 0;
      state.settings.model_version = document.getElementById('setting-model-version').value;
      modal.classList.add('hidden');
    });
  }
}

// Code Workspace Logic
function initCodeWorkspace() {
  const selector = document.getElementById('code-snippet-select');
  const codeEditor = document.getElementById('code-editor-textarea');
  const runBtn = document.getElementById('btn-run-code');
  const copyBtn = document.getElementById('btn-copy-code');
  const copyText = document.getElementById('copy-code-text');

  function updateCodeEditor(type) {
    const snippet = CODE_TEMPLATES[type];
    if (snippet && codeEditor) {
      codeEditor.value = snippet.code;
    }
  }

  if (selector) {
    selector.addEventListener('change', (e) => updateCodeEditor(e.target.value));
    updateCodeEditor('python');
  }

  if (copyBtn && codeEditor) {
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(codeEditor.value);
        if (copyText) copyText.textContent = "Copied! ✓";
        setTimeout(() => { if (copyText) copyText.textContent = "Copy Code"; }, 2000);
      } catch (err) {
        alert("Failed to copy: " + err.message);
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      const consoleOutput = document.getElementById('code-console-output');
      const audioContainer = document.getElementById('code-audio-result');
      if (consoleOutput) consoleOutput.textContent = "[...] Running execution request via Chatterbox engine...";
      runBtn.disabled = true;

      try {
        const res = await fetch('/api/code/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            code_type: selector.value,
            text: "Hello from Chatterbox Z.ai Code Workspace!",
            language: state.selectedLanguage,
            exaggeration: state.settings.exaggeration,
            cfg_weight: state.settings.cfg_weight
          })
        });

        const data = await res.json();
        if (consoleOutput) {
          consoleOutput.textContent = data.logs ? data.logs.join('\n') : JSON.stringify(data, null, 2);
        }

        if (data.audio_url && audioContainer) {
          audioContainer.innerHTML = `
            <div class="mt-2 p-3 rounded-xl bg-zinc-950/80 border border-indigo-500/30 flex items-center justify-between">
              <span class="text-xs text-zinc-300 font-mono">Rendered Output: ${data.duration}s</span>
              <audio controls src="${data.audio_url}" class="h-8 max-w-[200px]"></audio>
            </div>
          `;
        }
      } catch (err) {
        if (consoleOutput) consoleOutput.textContent = `[ERROR] Execution failed: ${err.message}`;
      } finally {
        runBtn.disabled = false;
      }
    });
  }
}

// Workers Workspace Logic
function initWorkers() {
  const batchSubmitBtn = document.getElementById('btn-submit-batch-worker');
  const dubbingSubmitBtn = document.getElementById('btn-submit-dubbing-worker');
  const clearJobsBtn = document.getElementById('btn-clear-jobs');

  if (clearJobsBtn) {
    clearJobsBtn.addEventListener('click', async () => {
      try {
        await fetch('/api/workers/jobs/clear', { method: 'POST' });
        refreshWorkersJobs();
      } catch (err) {
        console.error("Failed to clear jobs:", err);
      }
    });
  }

  if (batchSubmitBtn) {
    batchSubmitBtn.addEventListener('click', async () => {
      const textBlock = document.getElementById('batch-input-lines').value;
      const lang = document.getElementById('batch-language-select').value;
      const lines = textBlock.split('\n').map(l => l.trim()).filter(l => l.length > 0);

      if (lines.length === 0) {
        alert("Please enter at least one line of text.");
        return;
      }

      batchSubmitBtn.disabled = true;
      try {
        const res = await fetch('/api/workers/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            texts: lines,
            language: lang,
            ref_audio: state.referenceAudioUrl,
            exaggeration: state.settings.exaggeration,
            cfg_weight: state.settings.cfg_weight
          })
        });
        const data = await res.json();
        if (data.success) {
          refreshWorkersJobs();
          document.getElementById('batch-input-lines').value = '';
        }
      } catch (err) {
        alert('Error queuing batch job: ' + err.message);
      } finally {
        batchSubmitBtn.disabled = false;
      }
    });
  }

  if (dubbingSubmitBtn) {
    dubbingSubmitBtn.addEventListener('click', async () => {
      const sourceText = document.getElementById('dubbing-source-text').value.trim();
      const checkboxes = document.querySelectorAll('.dubbing-target-lang:checked');
      const targetLangs = Array.from(checkboxes).map(c => c.value);

      if (!sourceText) {
        alert("Please enter source text for dubbing.");
        return;
      }
      if (targetLangs.length === 0) {
        alert("Please select at least one target language.");
        return;
      }

      dubbingSubmitBtn.disabled = true;
      try {
        const res = await fetch('/api/workers/dubbing', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_text: sourceText,
            target_languages: targetLangs,
            ref_audio: state.referenceAudioUrl
          })
        });
        const data = await res.json();
        if (data.success) {
          refreshWorkersJobs();
        }
      } catch (err) {
        alert('Error queuing dubbing job: ' + err.message);
      } finally {
        dubbingSubmitBtn.disabled = false;
      }
    });
  }
}

async function refreshWorkersJobs() {
  try {
    const res = await fetch('/api/workers/jobs');
    const data = await res.json();
    const container = document.getElementById('workers-jobs-list');
    if (!container) return;

    if (!data.jobs || data.jobs.length === 0) {
      container.innerHTML = `
        <div class="text-center py-12 text-zinc-500 text-sm">
          No background worker jobs queued yet. Launch a batch or dubbing workflow above!
        </div>
      `;
      return;
    }

    container.innerHTML = data.jobs.map(job => {
      const isCompleted = job.status === 'completed';
      const isFailed = job.status === 'failed';
      const isRunning = job.status === 'running';

      let statusColor = 'bg-zinc-700 text-zinc-300';
      if (isCompleted) statusColor = 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
      if (isFailed) statusColor = 'bg-rose-500/20 text-rose-300 border border-rose-500/30';
      if (isRunning) statusColor = 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 animate-pulse';

      let filesHtml = '';
      if (job.result && job.result.files) {
        filesHtml = `
          <div class="mt-3 pt-3 border-t border-white/5 flex flex-wrap gap-2">
            ${job.result.files.map(f => `
              <div class="flex items-center gap-2 p-2 rounded-lg bg-zinc-950/70 border border-white/5 text-xs">
                <span class="font-mono text-zinc-400">#${f.index} (${f.duration}s)</span>
                <audio controls src="${f.url}" class="h-6 w-32"></audio>
                <a href="${f.url}" download class="text-indigo-400 hover:text-indigo-300">⬇</a>
              </div>
            `).join('')}
          </div>
        `;
      } else if (job.result && job.result.tracks) {
        filesHtml = `
          <div class="mt-3 pt-3 border-t border-white/5 flex flex-wrap gap-2">
            ${job.result.tracks.map(t => `
              <div class="flex items-center gap-2 p-2 rounded-lg bg-zinc-950/70 border border-white/5 text-xs">
                <span class="font-medium text-indigo-300">${t.language_name}</span>
                <audio controls src="${t.url}" class="h-6 w-32"></audio>
                <a href="${t.url}" download class="text-indigo-400 hover:text-indigo-300">⬇</a>
              </div>
            `).join('')}
          </div>
        `;
      }

      return `
        <div class="p-4 rounded-xl glass-panel border border-white/5 mb-3">
          <div class="flex items-center justify-between gap-2">
            <div class="flex items-center gap-2.5">
              <span class="text-xs px-2.5 py-1 rounded-full uppercase font-bold tracking-wider ${statusColor}">${job.status}</span>
              <h4 class="text-sm font-semibold text-white">${escapeHtml(job.title)}</h4>
            </div>
            <div class="flex items-center gap-2">
              <span class="text-xs font-mono text-zinc-500">${job.id}</span>
              <button onclick="deleteWorkerJob('${job.id}')" class="text-zinc-500 hover:text-rose-400 text-xs">✕</button>
            </div>
          </div>

          <!-- Progress Bar -->
          <div class="w-full bg-zinc-800/80 rounded-full h-1.5 mt-3 overflow-hidden">
            <div class="bg-gradient-to-r from-indigo-500 to-cyan-400 h-1.5 rounded-full transition-all duration-300" style="width: ${job.progress}%"></div>
          </div>

          <!-- Logs Accordion -->
          <div class="mt-3 text-xs font-mono text-zinc-400 bg-zinc-950/60 rounded-lg p-2.5 max-h-24 overflow-y-auto">
            ${job.logs.slice(-3).map(l => `<div>${escapeHtml(l)}</div>`).join('')}
          </div>

          ${filesHtml}
        </div>
      `;
    }).join('');
  } catch (err) {
    console.error("Error refreshing worker jobs:", err);
  }
}

async function deleteWorkerJob(jobId) {
  try {
    await fetch(`/api/workers/jobs/${jobId}`, { method: 'DELETE' });
    refreshWorkersJobs();
  } catch (err) {
    console.error("Failed to delete job:", err);
  }
}

// Billing & Scan to Pay Controller
let currentOrderId = null;
let qrTimerInterval = null;

function initBillingModal() {
  const modal = document.getElementById('billing-modal');
  const openBtn = document.getElementById('btn-open-billing');
  const closeBtn = document.getElementById('btn-close-billing');
  const backBtn = document.getElementById('btn-back-plans');
  const confirmBtn = document.getElementById('btn-confirm-payment');

  if (openBtn && modal) {
    openBtn.addEventListener('click', () => {
      showBillingPlansView();
      modal.classList.remove('hidden');
    });
  }

  if (closeBtn && modal) {
    closeBtn.addEventListener('click', () => {
      modal.classList.add('hidden');
      if (qrTimerInterval) clearInterval(qrTimerInterval);
    });
  }

  if (backBtn) {
    backBtn.addEventListener('click', showBillingPlansView);
  }

  if (confirmBtn) {
    confirmBtn.addEventListener('click', verifyScanPayment);
  }
}

async function loadBillingPlansAndBalance() {
  try {
    const [balanceRes, plansRes] = await Promise.all([
      fetch('/api/billing/balance'),
      fetch('/api/billing/plans')
    ]);

    const balanceData = await balanceRes.json();
    const plansData = await plansRes.json();

    updateCreditsDisplay(balanceData.credits);
    renderBillingPlans(plansData.plans || []);
  } catch (err) {
    console.error("Failed to load billing data:", err);
  }
}

function updateCreditsDisplay(credits) {
  const formatted = Number(credits).toLocaleString();
  const navDisplay = document.getElementById('user-credits-display');
  const modalDisplay = document.getElementById('modal-credits-display');
  if (navDisplay) navDisplay.textContent = formatted;
  if (modalDisplay) modalDisplay.textContent = formatted;
}

function renderBillingPlans(plans) {
  const container = document.getElementById('billing-plans-cards');
  if (!container) return;

  container.innerHTML = plans.map(p => `
    <div class="p-4 rounded-xl glass-panel border ${p.popular ? 'border-amber-500/50 bg-amber-500/5' : 'border-white/5'} flex flex-col justify-between relative group">
      ${p.popular ? `<span class="absolute -top-2.5 right-3 text-[9px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-amber-500 text-black">Popular</span>` : ''}
      <div>
        <h4 class="text-sm font-bold text-white mb-1">${escapeHtml(p.name)}</h4>
        <div class="text-lg font-black text-amber-400 font-mono">₹${p.price_inr} <span class="text-xs text-zinc-400 font-normal">($${p.price_usd})</span></div>
        <div class="text-[11px] text-emerald-400 font-medium mt-1">⚡ ${p.credits.toLocaleString()} Credits</div>
        <div class="text-[10px] text-zinc-400 mt-0.5">${p.characters_est}</div>
        
        <ul class="mt-3 space-y-1 text-[11px] text-zinc-400 border-t border-white/5 pt-2">
          ${p.features.map(f => `<li class="flex items-center gap-1.5"><span class="text-emerald-400 text-xs">✓</span> ${escapeHtml(f)}</li>`).join('')}
        </ul>
      </div>

      <button onclick="startScanToPay('${p.id}')" class="mt-4 w-full py-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-black font-bold text-xs transition shadow-md flex items-center justify-center gap-1.5">
        <span>📲</span> Scan to Pay
      </button>
    </div>
  `).join('');
}

function showBillingPlansView() {
  document.getElementById('billing-plans-view').classList.remove('hidden');
  document.getElementById('billing-qr-view').classList.add('hidden');
  document.getElementById('billing-qr-view').classList.remove('flex');
  if (qrTimerInterval) clearInterval(qrTimerInterval);
}

async function startScanToPay(planId) {
  try {
    const res = await fetch('/api/billing/create-order', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan_id: planId })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.detail || "Failed to create order");

    const order = data.order;
    currentOrderId = order.order_id;

    // Switch view to QR
    document.getElementById('billing-plans-view').classList.add('hidden');
    const qrView = document.getElementById('billing-qr-view');
    qrView.classList.remove('hidden');
    qrView.classList.add('flex');

    // Update Details
    document.getElementById('qr-plan-title').textContent = `${order.plan_name} (${order.credits.toLocaleString()} Credits)`;
    document.getElementById('qr-amount-display').textContent = `₹${order.amount_inr} / $${order.amount_usd}`;
    document.getElementById('qr-order-id').textContent = order.order_id;

    // Render Dynamic QR SVG
    renderQrCodeSvg(order.upi_url);

    // Start 15-minute countdown
    startQrCountdown(order.expires_in_seconds || 900);
  } catch (err) {
    alert("Scan to Pay error: " + err.message);
  }
}

function renderQrCodeSvg(payload) {
  const svg = document.getElementById('qr-svg-canvas');
  if (!svg) return;

  // Generate clean 21x21 QR pattern with finder squares and matrix payload
  const size = 21;
  const cellSize = 8;
  const offset = 16;
  let rects = [];

  // Corner Finder Patterns (Top-Left, Top-Right, Bottom-Left)
  const addFinder = (x, y) => {
    rects.push(`<rect x="${x}" y="${y}" width="56" height="56" fill="black" rx="4"/>`);
    rects.push(`<rect x="${x + 8}" y="${y + 8}" width="40" height="40" fill="white" rx="2"/>`);
    rects.push(`<rect x="${x + 16}" y="${y + 16}" width="24" height="24" fill="black" rx="2"/>`);
  };

  addFinder(offset, offset);
  addFinder(offset + 112, offset);
  addFinder(offset, offset + 112);

  // Deterministic pseudo-random payload grid based on payload string hash
  let hash = 0;
  for (let i = 0; i < payload.length; i++) {
    hash = ((hash << 5) - hash) + payload.charCodeAt(i);
    hash |= 0;
  }

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      // Avoid finder pattern zones
      if ((r < 8 && c < 8) || (r < 8 && c > 13) || (r > 13 && c < 8)) continue;
      // Deterministic bit fill
      const bit = Math.abs((hash ^ (r * 31 + c * 17))) % 3 === 0;
      if (bit) {
        rects.push(`<rect x="${offset + c * cellSize}" y="${offset + r * cellSize}" width="${cellSize - 1}" height="${cellSize - 1}" fill="black" rx="1"/>`);
      }
    }
  }

  // Add Center Branding Dot
  rects.push(`<circle cx="100" cy="100" r="14" fill="#6366f1"/>`);
  rects.push(`<text x="100" y="104" font-size="10" font-weight="bold" fill="white" text-anchor="middle" font-family="sans-serif">Z</text>`);

  svg.innerHTML = rects.join('');
}

function startQrCountdown(duration) {
  if (qrTimerInterval) clearInterval(qrTimerInterval);
  let timeLeft = duration;
  const timerDisplay = document.getElementById('qr-timer');

  const update = () => {
    const mins = Math.floor(timeLeft / 60);
    const secs = timeLeft % 60;
    if (timerDisplay) {
      timerDisplay.textContent = `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    }
    if (timeLeft <= 0) {
      clearInterval(qrTimerInterval);
      if (timerDisplay) timerDisplay.textContent = "Expired";
    }
    timeLeft--;
  };

  update();
  qrTimerInterval = setInterval(update, 1000);
}

async function verifyScanPayment() {
  if (!currentOrderId) return;
  const btn = document.getElementById('btn-confirm-payment');
  if (btn) btn.innerHTML = `<span class="animate-spin inline-block mr-1">↻</span> Verifying payment...`;

  try {
    const res = await fetch('/api/billing/verify-payment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ order_id: currentOrderId })
    });

    const data = await res.json();
    if (!data.success) throw new Error(data.detail || "Payment not verified yet.");

    // Success notification
    updateCreditsDisplay(data.new_balance);
    alert(`🎉 Payment Confirmed!\n${data.message}\nYour new balance is ${data.new_balance.toLocaleString()} credits.`);

    // Close modal
    const modal = document.getElementById('billing-modal');
    if (modal) modal.classList.add('hidden');
    if (qrTimerInterval) clearInterval(qrTimerInterval);
  } catch (err) {
    alert("Payment verification: " + err.message);
  } finally {
    if (btn) btn.innerHTML = `<span>✓</span> I Have Paid • Verify`;
  }
}

// Helpers
function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>"']/g, m => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  })[m]);
}

