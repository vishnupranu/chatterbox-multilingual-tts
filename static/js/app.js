/**
 * app.js
 * Frontend controller for Chatterbox Z.ai Multilingual Platform.
 * Manages Chat, Code workspace, autonomous Workers, live mic recording, and voice library.
 */

// Application State
const state = {
  currentTab: 'landing',
  selectedLanguage: 'en',
  selectedModel: 'local_neural',
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
  skills: [],
  models: [],
  connectors: [],
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
  initHeaderModelSelector();
  await loadSystemStatusAndLanguages();
  await loadPresetVoices();
  await loadBillingPlansAndBalance();
  await refreshSkillsList();
  await loadSkillsMarketplace();
  await loadTeamRoster();
  await loadLLMModelsMatrix();
  await loadConnectorsList();
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

    // Populate Landing Page Interactive Showcase
    renderLandingVoices();
  } catch (err) {
    console.error("Failed to load preset voices:", err);
  }
}

function renderLandingVoices() {
  const grid = document.getElementById('landing-voices-grid');
  if (!grid) return;

  const voices = (state.presets && state.presets.length > 0) ? state.presets : [
    { id: 'en_female', name: 'Sarah', lang: 'en', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac', avatar: '👩🏼', desc: 'English (US)' },
    { id: 'fr_female', name: 'Camille', lang: 'fr', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac', avatar: '👩🏻', desc: 'French (FR)' },
    { id: 'es_female', name: 'Elena', lang: 'es', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac', avatar: '👩🏽', desc: 'Spanish (ES)' },
    { id: 'de_female', name: 'Greta', lang: 'de', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac', avatar: '👱🏻‍♀️', desc: 'German (DE)' },
    { id: 'hi_female', name: 'Aanya', lang: 'hi', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac', avatar: '👩🏾', desc: 'Hindi (IN)' },
    { id: 'ja_female', name: 'Yuki', lang: 'ja', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac', avatar: '👧🏻', desc: 'Japanese (JP)' },
    { id: 'zh_female', name: 'Mei', lang: 'zh', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac', avatar: '👩🏻', desc: 'Chinese (Mandarin)' },
    { id: 'ru_male', name: 'Dmitri', lang: 'ru', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac', avatar: '👨🏼', desc: 'Russian (RU)' },
    { id: 'it_male', name: 'Marco', lang: 'it', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac', avatar: '👨🏻', desc: 'Italian (IT)' },
    { id: 'pt_male', name: 'Thiago', lang: 'pt', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac', avatar: '👨🏽', desc: 'Portuguese (BR)' },
    { id: 'ar_female', name: 'Layla', lang: 'ar', audio: 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac', avatar: '🧕🏽', desc: 'Arabic (MSA)' }
  ];

  grid.innerHTML = voices.map(v => `
    <div class="voice-sample-chip p-3 rounded-xl flex flex-col justify-between gap-2">
      <div class="flex items-center justify-between">
        <span class="text-[11px] font-mono font-bold px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300">${v.lang.toUpperCase()}</span>
        <button onclick="playLandingVoiceAudio('${v.audio}', this)" class="w-7 h-7 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center transition shadow-md" title="Audition Voice">
          <svg class="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
        </button>
      </div>
      <div>
        <div class="text-xs font-bold text-white truncate">${v.name}</div>
        <div class="text-[10px] text-zinc-400 capitalize truncate">${v.desc || v.lang}</div>
      </div>
      <button onclick="selectPresetVoice('${v.id}'); switchWorkspace('chat');" class="w-full py-1 rounded-lg bg-white/5 hover:bg-indigo-600/30 text-[10px] text-zinc-300 hover:text-white transition">
        Use Voice →
      </button>
    </div>
  `).join('');
}
window.renderLandingVoices = renderLandingVoices;

let currentLandingAudio = null;
function playLandingVoiceAudio(url, btn) {
  if (currentLandingAudio) {
    currentLandingAudio.pause();
    currentLandingAudio = null;
    document.querySelectorAll('#landing-voices-grid button svg').forEach(svg => {
      svg.parentElement.innerHTML = '<svg class="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
    });
  }
  const audio = new Audio(url);
  currentLandingAudio = audio;
  btn.innerHTML = '<svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
  audio.play();
  audio.onended = () => {
    btn.innerHTML = '<svg class="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
    currentLandingAudio = null;
  };
}
window.playLandingVoiceAudio = playLandingVoiceAudio;

function openScanToPayForPlan(planId) {
  const modal = document.getElementById('billing-modal');
  if (modal) modal.classList.remove('hidden');
  selectBillingPlan(planId);
}
window.openScanToPayForPlan = openScanToPayForPlan;

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
  const tabs = ['landing', 'team', 'chat', 'skills', 'models', 'connectors', 'code', 'workers'];
  tabs.forEach(tabName => {
    const btn = document.getElementById(`nav-btn-${tabName}`);
    if (btn) {
      btn.addEventListener('click', () => switchTab(tabName));
    }
  });
}

const TAB_TITLES = {
  landing: "Chatterbox GS • Multilingual Flow-Matching Showcase",
  team: "Human AI Characters & Team • GS Leadership & Acoustic Personas",
  chat: "Chat Assistant • 23 Languages & Zero-Shot Voice Cloning",
  skills: "Antigravity Autonomous Agent Skills Studio • Multi-Voice & Cultural Dubbing",
  models: "Multi-Model LLM Matrix • Frontier AI Reasoning & Speech",
  connectors: "Connectors & Ecosystem • Provider Keys & Inbound Webhooks",
  code: "Interactive Code Runner & REST API Sandbox",
  workers: "Autonomous Background Workers & Mass Dubbing Pipeline"
};

function switchTab(targetTab) {
  state.currentTab = targetTab;
  const tabs = ['landing', 'team', 'chat', 'skills', 'models', 'connectors', 'code', 'workers'];

  const topTitle = document.getElementById('top-title');
  if (topTitle && TAB_TITLES[targetTab]) {
    topTitle.textContent = TAB_TITLES[targetTab];
  }

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
        if (tab === 'landing') {
          view.classList.add('block');
        } else {
          view.classList.add('flex');
        }
        view.classList.add('animate-fade-in');
      } else {
        view.classList.add('hidden');
        view.classList.remove('flex', 'block', 'animate-fade-in');
      }
    }
  });

  if (targetTab === 'team') {
    loadTeamRoster();
  } else if (targetTab === 'skills') {
    loadSkillsMarketplace();
  }

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
window.switchWorkspace = switchTab;

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
    let data;
    if (state.selectedModel && state.selectedModel !== 'local_neural') {
      const res = await fetch('/api/llm/chat-and-speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: text,
          model_id: state.selectedModel,
          language: lang,
          audio_prompt_url: state.referenceAudioUrl
        })
      });
      data = await res.json();
      loadingDiv.remove();
      if (!res.ok) {
        throw new Error(data.detail || "LLM Chat request failed");
      }
      addChatMessage({
        role: 'assistant',
        text: `[${data.model.toUpperCase()} • ${data.provider}]: ${data.text}`,
        language: data.language || lang,
        audio_url: data.audio_url,
        duration: data.duration || 0,
      });
    } else {
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

      data = await res.json();
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
    }
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
    const upiDisplay = document.getElementById('qr-upi-id-display');
    if (upiDisplay && order.pay_address) {
      upiDisplay.textContent = order.pay_address;
    }

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

// ==============================================================================
// AITalk Project-Trained Copilot Controller
// ==============================================================================
function toggleAITalkDrawer() {
  const drawer = document.getElementById('aitalk-drawer');
  if (drawer) {
    drawer.classList.toggle('open');
    if (drawer.classList.contains('open')) {
      document.getElementById('aitalk-input')?.focus();
    }
  }
}
window.toggleAITalkDrawer = toggleAITalkDrawer;

async function sendAITalkQuery(query) {
  const messagesContainer = document.getElementById('aitalk-messages');
  if (!messagesContainer || !query.trim()) return;

  // Open drawer if closed
  const drawer = document.getElementById('aitalk-drawer');
  if (drawer && !drawer.classList.contains('open')) {
    drawer.classList.add('open');
  }

  // Append user message
  const userMsgEl = document.createElement('div');
  userMsgEl.className = 'p-3 rounded-xl bg-indigo-600/20 border border-indigo-500/30 text-white ml-6 text-xs';
  userMsgEl.innerHTML = `<div class="font-semibold text-indigo-300 text-[10px] mb-1">YOU</div><div>${escapeHtml(query)}</div>`;
  messagesContainer.appendChild(userMsgEl);

  // Append thinking bubble
  const aiMsgEl = document.createElement('div');
  aiMsgEl.className = 'p-3.5 rounded-xl bg-zinc-900/90 border border-white/10 text-zinc-300 mr-4 text-xs space-y-2';
  aiMsgEl.innerHTML = `
    <div class="font-semibold text-purple-400 text-[10px] flex items-center gap-1.5">
      <span class="w-1.5 h-1.5 rounded-full bg-purple-400 animate-ping"></span>
      AITALK COPILOT
    </div>
    <div class="text-zinc-400 italic">Project context retrieval in progress...</div>
  `;
  messagesContainer.appendChild(aiMsgEl);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  try {
    const res = await fetch('/api/copilot/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, context: state.currentTab })
    });
    const data = await res.json();

    if (data.success) {
      let actionsHtml = '';
      if (data.suggested_actions && data.suggested_actions.length > 0) {
        actionsHtml = `
          <div class="pt-2 flex flex-wrap gap-1.5 border-t border-white/5">
            ${data.suggested_actions.map(act => `
              <button onclick="handleCopilotAction('${encodeURIComponent(JSON.stringify(act))}')" class="px-2 py-1 rounded-md bg-indigo-500/20 hover:bg-indigo-500/30 border border-indigo-500/30 text-[10px] font-semibold text-indigo-300 transition">
                ⚡ ${act.label}
              </button>
            `).join('')}
          </div>
        `;
      }

      // Convert simple markdown code blocks and headers to HTML
      let formatted = data.answer
        .replace(/### (.*)/g, '<h4 class="font-bold text-white text-xs mt-1 mb-1">$1</h4>')
        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>')
        .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-black/40 text-amber-300 font-mono text-[10px]">$1</code>')
        .replace(/```python([\s\S]*?)```/g, '<pre class="bg-black/60 p-2.5 rounded-lg font-mono text-[10px] text-emerald-300 overflow-x-auto my-2 border border-white/5">$1</pre>')
        .replace(/```bash([\s\S]*?)```/g, '<pre class="bg-black/60 p-2.5 rounded-lg font-mono text-[10px] text-cyan-300 overflow-x-auto my-2 border border-white/5">$1</pre>')
        .replace(/\n\n/g, '<br><br>');

      aiMsgEl.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="font-semibold text-purple-400 text-[10px] flex items-center gap-1.5">
            <span>✨</span> AITALK COPILOT
          </div>
          <button onclick="speakAITalkText(this)" data-text="${escapeHtml(data.answer.replace(/#|\*|`|```[\s\S]*?```/g, ''))}" class="text-[10px] text-zinc-400 hover:text-white flex items-center gap-1 bg-white/5 px-2 py-0.5 rounded transition">
            <span>🔊</span> Listen
          </button>
        </div>
        <div class="leading-relaxed text-zinc-200">${formatted}</div>
        ${actionsHtml}
      `;
    } else {
      aiMsgEl.innerHTML = `<div class="text-rose-400">Error: ${data.detail || 'Could not answer query'}</div>`;
    }
  } catch (err) {
    aiMsgEl.innerHTML = `<div class="text-rose-400">Network error: ${err.message}</div>`;
  }
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
window.sendAITalkQuery = sendAITalkQuery;

function submitAITalkInput() {
  const input = document.getElementById('aitalk-input');
  if (input && input.value.trim()) {
    const val = input.value.trim();
    input.value = '';
    sendAITalkQuery(val);
  }
}
window.submitAITalkInput = submitAITalkInput;

function handleCopilotAction(encodedAct) {
  const act = JSON.parse(decodeURIComponent(encodedAct));
  if (act.type === 'switch_tab') {
    switchWorkspace(act.target);
  } else if (act.type === 'open_modal') {
    if (act.target === 'presetModal') openPresetsModal();
    if (act.target === 'paymentModal') openBillingModal();
  } else if (act.type === 'chat_fill') {
    switchWorkspace('chat');
    const input = document.getElementById('chat-input-text');
    if (input) input.value = act.text;
    if (act.lang) {
      const select = document.getElementById('chat-language-select');
      if (select) select.value = act.lang;
      state.selectedLanguage = act.lang;
    }
  } else if (act.type === 'copilot_ask') {
    sendAITalkQuery(act.query);
  }
}
window.handleCopilotAction = handleCopilotAction;

function speakAITalkText(btn) {
  const text = btn.getAttribute('data-text');
  if (!text) return;
  btn.innerHTML = `<span>⏳</span> Synthesizing...`;
  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text.slice(0, 250), language: 'en', model_version: 'v3' })
  })
  .then(r => r.json())
  .then(data => {
    if (data.success && data.audio_url) {
      const audio = new Audio(data.audio_url);
      audio.play();
      btn.innerHTML = `<span>🔊</span> Playing`;
      audio.onended = () => { btn.innerHTML = `<span>🔊</span> Listen`; };
    } else {
      btn.innerHTML = `<span>🔊</span> Listen`;
    }
  })
  .catch(() => { btn.innerHTML = `<span>🔊</span> Listen`; });
}
window.speakAITalkText = speakAITalkText;

// ==============================================================================
// Header LLM Model Selector
// ==============================================================================
function initHeaderModelSelector() {
  const selector = document.getElementById('header-llm-model-select');
  if (selector) {
    selector.value = state.selectedModel || 'local_neural';
    selector.addEventListener('change', (e) => {
      state.selectedModel = e.target.value;
      console.log(`[Chatterbox] Switched active reasoning model to: ${state.selectedModel}`);
    });
  }
}
window.initHeaderModelSelector = initHeaderModelSelector;

// ==============================================================================
// Autonomous Agent Skills Studio Controller
// ==============================================================================
async function refreshSkillsList() {
  try {
    const res = await fetch('/api/skills/list');
    const data = await res.json();
    state.skills = data.skills || [];
    console.log(`[Skills] Loaded ${state.skills.length} autonomous agent skills.`);
  } catch (err) {
    console.error("[Skills] Failed to fetch skills:", err);
  }
}
window.refreshSkillsList = refreshSkillsList;

// 1. Podcast Studio Producer Skill
async function executePodcastSkill() {
  const topic = document.getElementById('skill-podcast-topic')?.value.trim() || "Advances in Neural Flow Matching";
  const turns = parseInt(document.getElementById('skill-podcast-turns')?.value || '3');
  const llm = document.getElementById('skill-podcast-llm')?.value || 'local_neural';
  const btn = document.getElementById('btn-skill-run-podcast');
  const resContainer = document.getElementById('skill-podcast-result');

  if (!resContainer) return;
  resContainer.classList.remove('hidden');
  resContainer.innerHTML = `
    <div class="p-3.5 rounded-xl bg-zinc-950/90 border border-purple-500/30 text-xs text-purple-300 flex items-center gap-2">
      <svg class="animate-spin w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
      Generating dual-host podcast dialog and recording alternating 24kHz tracks...
    </div>
  `;
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/skills/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        skill_id: 'podcast_host',
        params: { topic: topic, turns_count: turns, model_id: llm }
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Podcast generation failed");

    let dialogHtml = '';
    if (data.dialog && data.dialog.length > 0) {
      dialogHtml = data.dialog.map(d => `
        <div class="p-2.5 rounded-lg ${d.speaker === 'Host A' ? 'bg-purple-950/40 border border-purple-500/20' : 'bg-indigo-950/40 border border-indigo-500/20'} text-xs">
          <span class="font-bold font-mono text-[10px] ${d.speaker === 'Host A' ? 'text-purple-400' : 'text-indigo-400'}">${d.speaker}</span>: 
          <span class="text-zinc-200">${escapeHtml(d.text)}</span>
        </div>
      `).join('');
    }

    resContainer.innerHTML = `
      <div class="p-4 rounded-xl bg-zinc-950/90 border border-purple-500/30 space-y-3">
        <div class="flex items-center justify-between text-xs border-b border-white/10 pb-2">
          <div class="font-bold text-white flex items-center gap-2">
            <span>🎙️</span> ${escapeHtml(data.episode_title || topic)}
          </div>
          <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">${data.duration_est || '~15s'}</span>
        </div>
        <div class="space-y-1.5 max-h-48 overflow-y-auto pr-1">
          ${dialogHtml}
        </div>
        ${data.audio_url ? `
          <div class="pt-2 border-t border-white/10 flex items-center justify-between gap-3">
            <audio controls src="${data.audio_url}" class="w-full h-8"></audio>
            <a href="${data.audio_url}" download="podcast_episode.wav" class="px-2.5 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs flex-shrink-0 transition">
              Download WAV
            </a>
          </div>
        ` : ''}
      </div>
    `;
  } catch (err) {
    resContainer.innerHTML = `<div class="p-3 rounded-xl bg-rose-950/50 border border-rose-500/30 text-xs text-rose-300">Error: ${err.message}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}
window.executePodcastSkill = executePodcastSkill;

// 2. Cross-Lingual Cultural Dubber Skill
async function executeDubbingSkill() {
  const text = document.getElementById('skill-dub-text')?.value.trim() || "Welcome to Chatterbox multilingual synthesis.";
  const lang = document.getElementById('skill-dub-lang')?.value || "ja";
  const btn = document.getElementById('btn-skill-run-dub');
  const resContainer = document.getElementById('skill-dub-result');

  if (!resContainer) return;
  resContainer.classList.remove('hidden');
  resContainer.innerHTML = `
    <div class="p-3.5 rounded-xl bg-zinc-950/90 border border-cyan-500/30 text-xs text-cyan-300 flex items-center gap-2">
      <svg class="animate-spin w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
      Translating culturally into ${lang.toUpperCase()} and synthesizing with matching timbre...
    </div>
  `;
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/skills/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        skill_id: 'cross_lingual_translator',
        params: { source_text: text, target_language: lang }
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Dubbing failed");

    resContainer.innerHTML = `
      <div class="p-4 rounded-xl bg-zinc-950/90 border border-cyan-500/30 space-y-3">
        <div class="flex items-center justify-between text-xs border-b border-white/10 pb-2">
          <span class="font-bold text-white flex items-center gap-2">
            <span>🌐</span> Target: ${data.target_language.toUpperCase()}
          </span>
          <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">${data.duration ? data.duration + 's' : 'Ready'}</span>
        </div>
        <div class="text-xs p-2.5 rounded-lg bg-cyan-950/30 border border-cyan-500/20 text-cyan-200">
          ${escapeHtml(data.translated_text)}
        </div>
        ${data.audio_url ? `
          <div class="pt-2 border-t border-white/10 flex items-center justify-between gap-3">
            <audio controls src="${data.audio_url}" class="w-full h-8"></audio>
            <a href="${data.audio_url}" download="dubbed_${lang}.wav" class="px-2.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs flex-shrink-0 transition">
              Download WAV
            </a>
          </div>
        ` : ''}
      </div>
    `;
  } catch (err) {
    resContainer.innerHTML = `<div class="p-3 rounded-xl bg-rose-950/50 border border-rose-500/30 text-xs text-rose-300">Error: ${err.message}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}
window.executeDubbingSkill = executeDubbingSkill;

// 3. Document Narrator Skill
async function executeNarratorSkill() {
  const doc = document.getElementById('skill-narrator-text')?.value.trim() || "Chapter 1: The Acoustic Horizon.";
  const style = document.getElementById('skill-narrator-style')?.value || "Warm Conversational";
  const btn = document.getElementById('btn-skill-run-narrator');
  const resContainer = document.getElementById('skill-narrator-result');

  if (!resContainer) return;
  resContainer.classList.remove('hidden');
  resContainer.innerHTML = `
    <div class="p-3.5 rounded-xl bg-zinc-950/90 border border-amber-500/30 text-xs text-amber-300 flex items-center gap-2">
      <svg class="animate-spin w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
      Analyzing paragraph flow and narrating chapter audio...
    </div>
  `;
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/skills/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        skill_id: 'document_narrator',
        params: { document_text: doc, voice_style: style }
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Narration failed");

    resContainer.innerHTML = `
      <div class="p-4 rounded-xl bg-zinc-950/90 border border-amber-500/30 space-y-3">
        <div class="flex items-center justify-between text-xs border-b border-white/10 pb-2">
          <span class="font-bold text-white flex items-center gap-2">
            <span>📖</span> Style: ${data.style || style}
          </span>
          <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">${data.word_count || 0} Words</span>
        </div>
        ${data.audio_url ? `
          <div class="pt-2 border-t border-white/10 flex items-center justify-between gap-3">
            <audio controls src="${data.audio_url}" class="w-full h-8"></audio>
            <a href="${data.audio_url}" download="narrated_chapter.wav" class="px-2.5 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-semibold text-xs flex-shrink-0 transition">
              Download WAV
            </a>
          </div>
        ` : ''}
      </div>
    `;
  } catch (err) {
    resContainer.innerHTML = `<div class="p-3 rounded-xl bg-rose-950/50 border border-rose-500/30 text-xs text-rose-300">Error: ${err.message}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}
window.executeNarratorSkill = executeNarratorSkill;

// 4. Voice Director Skill
async function executeDirectorSkill() {
  const script = document.getElementById('skill-director-text')?.value.trim() || "Listen closely, the future of voice has arrived.";
  const mood = document.getElementById('skill-director-mood')?.value || "Dramatic Suspense";
  const btn = document.getElementById('btn-skill-run-director');
  const resContainer = document.getElementById('skill-director-result');

  if (!resContainer) return;
  resContainer.classList.remove('hidden');
  resContainer.innerHTML = `
    <div class="p-3.5 rounded-xl bg-zinc-950/90 border border-emerald-500/30 text-xs text-emerald-300 flex items-center gap-2">
      <svg class="animate-spin w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
      Directing acoustic emotion parameters (${mood}) and rendering audio...
    </div>
  `;
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/skills/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        skill_id: 'voice_director',
        params: { script: script, mood: mood }
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Voice direct failed");

    const tuning = data.tuning_applied || {};
    resContainer.innerHTML = `
      <div class="p-4 rounded-xl bg-zinc-950/90 border border-emerald-500/30 space-y-3">
        <div class="flex items-center justify-between text-xs border-b border-white/10 pb-2">
          <span class="font-bold text-white flex items-center gap-2">
            <span>🎬</span> Mood: ${data.mood || mood}
          </span>
          <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">CFG: ${tuning.cfg_weight || 0.5} • Exag: ${tuning.exaggeration || 0.5}</span>
        </div>
        ${data.audio_url ? `
          <div class="pt-2 border-t border-white/10 flex items-center justify-between gap-3">
            <audio controls src="${data.audio_url}" class="w-full h-8"></audio>
            <a href="${data.audio_url}" download="directed_scene.wav" class="px-2.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs flex-shrink-0 transition">
              Download WAV
            </a>
          </div>
        ` : ''}
      </div>
    `;
  } catch (err) {
    resContainer.innerHTML = `<div class="p-3 rounded-xl bg-rose-950/50 border border-rose-500/30 text-xs text-rose-300">Error: ${err.message}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}
window.executeDirectorSkill = executeDirectorSkill;

// 5. Khyathi.Sri Kids Songs & Story Studio Skill
async function executeKidsSkill() {
  const theme = document.getElementById('skill-kids-theme')?.value.trim() || "Rainbow Butterfly in Sunny Garden";
  const type = document.getElementById('skill-kids-type')?.value || "Nursery Rhyme Song";
  const btn = document.getElementById('btn-skill-run-kids');
  const resContainer = document.getElementById('skill-kids-result');

  if (!resContainer) return;
  resContainer.classList.remove('hidden');
  resContainer.innerHTML = `
    <div class="p-3.5 rounded-xl bg-zinc-950/90 border border-pink-500/30 text-xs text-pink-300 flex items-center gap-2">
      <svg class="animate-spin w-4 h-4 text-pink-400" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
      Khyathi.Sri is composing your song, generating colorful artwork, and recording 24kHz audio...
    </div>
  `;
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/skills/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        skill_id: 'kids_rhymes',
        params: { theme: theme, content_type: type, language: 'en' }
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Kids studio failed");

    resContainer.innerHTML = `
      <div class="p-4 rounded-xl bg-zinc-950/90 border border-pink-500/30 space-y-3">
        <div class="flex items-center justify-between text-xs border-b border-white/10 pb-2">
          <div class="flex items-center gap-2 text-white font-bold">
            <img src="/static/img/avatar_khyathi_sri.png" alt="Khyathi Sri" class="w-6 h-6 rounded-full object-cover border border-pink-400">
            <span>Khyathi.Sri • ${escapeHtml(data.content_type || type)}</span>
          </div>
          <div class="flex items-center gap-1.5 text-[10px] text-amber-300 font-mono">
            <img src="/static/img/partner_key_secure_foundation.png" class="w-4 h-4 object-contain">
            <span>Key Secure Verified</span>
          </div>
        </div>

        ${data.image_url ? `
          <div class="rounded-xl overflow-hidden border border-white/10 shadow-lg">
            <img src="${data.image_url}" alt="${escapeHtml(data.theme || '')}" class="w-full h-48 object-cover">
          </div>
        ` : ''}

        <p class="text-xs text-zinc-200 leading-relaxed italic bg-black/40 p-3 rounded-xl border border-white/5">
          "${escapeHtml(data.text || '')}"
        </p>

        ${data.audio_url ? `
          <div class="pt-2 border-t border-white/10 flex items-center justify-between gap-3">
            <audio controls src="${data.audio_url}" class="w-full h-8"></audio>
            <a href="${data.audio_url}" download="khyathi_kids_song.wav" class="px-3 py-1.5 rounded-lg bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white font-semibold text-xs flex-shrink-0 transition">
              Download WAV
            </a>
          </div>
        ` : ''}
      </div>
    `;
  } catch (err) {
    resContainer.innerHTML = `<div class="p-3 rounded-xl bg-rose-950/50 border border-rose-500/30 text-xs text-rose-300">Error: ${err.message}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}
window.executeKidsSkill = executeKidsSkill;

// ==============================================================================
// GS Human & AI Team Roster Controller
// ==============================================================================
async function loadTeamRoster() {
  const grid = document.getElementById('team-roster-grid');
  if (!grid) return;

  try {
    const res = await fetch('/api/team');
    const data = await res.json();
    state.teamMembers = data.team || [];

    grid.innerHTML = state.teamMembers.map(m => {
      const isFounder = m.id === 'char_founder';
      return `
        <div class="glass-panel rounded-2xl p-5 border border-white/5 hover:border-indigo-500/40 transition flex flex-col justify-between card-3d-character group">
          <div class="space-y-3">
            <div class="flex items-start justify-between gap-2">
              <div class="relative avatar-neural-glow">
                <img src="${m.avatar}" alt="${escapeHtml(m.name)}" class="w-14 h-14 rounded-2xl object-cover border border-white/20 shadow-md ${isFounder ? 'animate-character-float' : ''}">
                <span class="absolute -bottom-1 -right-1 w-3 h-3 rounded-full bg-emerald-400 border-2 border-[#0c0e15]"></span>
              </div>
              <span class="text-[9px] px-2 py-0.5 rounded-full font-mono font-bold uppercase border ${m.tag_color || 'border-indigo-500/40 text-indigo-300 bg-indigo-500/10'}">
                ${escapeHtml(m.tag || 'AGENT')}
              </span>
            </div>

            <div>
              <h4 class="text-sm font-bold text-white group-hover:text-indigo-300 transition">${escapeHtml(m.name)}</h4>
              <div class="text-[11px] text-zinc-400 font-medium">${escapeHtml(m.title)}</div>
              <div class="text-[10px] text-zinc-500 font-mono">${escapeHtml(m.role)}</div>
            </div>

            <p class="text-xs text-zinc-300 leading-relaxed line-clamp-3">
              ${escapeHtml(m.bio)}
            </p>

            <div class="flex flex-wrap gap-1 pt-1">
              ${(m.specialties || []).map(s => `
                <span class="text-[9px] px-2 py-0.5 rounded bg-white/5 text-zinc-400 font-mono border border-white/5">${escapeHtml(s)}</span>
              `).join('')}
            </div>
          </div>

          <div class="pt-4 mt-3 border-t border-white/5 space-y-2">
            <button onclick="auditionTeamVoice('${m.id}', this)" class="w-full py-2 rounded-xl bg-white/5 hover:bg-white/10 text-xs text-zinc-200 font-semibold transition flex items-center justify-center gap-1.5">
              <svg class="w-3.5 h-3.5 text-amber-400" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
              <span>Audition Voice</span>
            </button>
            <div class="grid grid-cols-2 gap-2">
              <button onclick="adoptTeamVoicePersona('${m.id}')" class="py-1.5 rounded-xl bg-white/5 hover:bg-indigo-600/20 text-[10px] text-zinc-300 font-medium border border-white/5 transition text-center">
                🎭 Clone Voice
              </button>
              <button onclick="chatWithTeamPersona('${m.id}')" class="py-1.5 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 text-[10px] text-indigo-300 font-medium border border-indigo-500/30 transition text-center">
                💬 Chat
              </button>
            </div>
          </div>
        </div>
      `;
    }).join('');
  } catch (err) {
    console.error("[Team] Failed to load team roster:", err);
  }
}
window.loadTeamRoster = loadTeamRoster;

async function auditionTeamVoice(memberId, btn) {
  const originalText = btn ? btn.innerHTML : '';
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<svg class="animate-spin w-3.5 h-3.5 text-amber-400 inline" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg> Auditioning...`;
  }

  try {
    const res = await fetch('/api/team/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ member_id: memberId })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Audition failed");

    const playerBox = document.getElementById('team-audio-player-box');
    const speakerLabel = document.getElementById('team-audio-speaker-label');
    const quoteText = document.getElementById('team-audio-quote-text');
    const durationLabel = document.getElementById('team-audio-duration');
    const audioElement = document.getElementById('team-audio-element');

    if (playerBox && audioElement) {
      playerBox.classList.remove('hidden');
      if (speakerLabel) speakerLabel.innerHTML = `<span>🔊</span> ${escapeHtml(data.member)} Voice Audition`;
      if (quoteText) quoteText.textContent = `"${data.text}"`;
      if (durationLabel) durationLabel.textContent = `Duration: ~${data.duration || 3.5}s`;
      audioElement.src = data.audio_url;
      audioElement.play().catch(() => {});
      playerBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  } catch (err) {
    alert(`Audition error: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = originalText;
    }
  }
}
window.auditionTeamVoice = auditionTeamVoice;

function adoptTeamVoicePersona(memberId) {
  const member = (state.teamMembers || []).find(m => m.id === memberId) || { name: "G.S. Founder", preset_voice: "ru_male" };
  if (member.preset_voice) {
    selectPresetVoice(member.preset_voice);
  }
  const badge = document.getElementById('voice-clone-badge');
  const filenameEl = document.getElementById('voice-clone-filename');
  if (badge && filenameEl) {
    badge.classList.remove('hidden');
    badge.classList.add('flex');
    filenameEl.textContent = `${member.name} Persona`;
  }
  alert(`Active voice timbre set to ${member.name}! Ready for zero-shot cloning.`);
}
window.adoptTeamVoicePersona = adoptTeamVoicePersona;

function chatWithTeamPersona(memberId) {
  adoptTeamVoicePersona(memberId);
  switchTab('chat');
  const chatInput = document.getElementById('chat-input-text');
  const member = (state.teamMembers || []).find(m => m.id === memberId) || { name: "G.S." };
  if (chatInput) {
    chatInput.value = `Hello ${member.name}, I would love to test your voice flow-matching and neural speech model!`;
    chatInput.focus();
  }
}
window.chatWithTeamPersona = chatWithTeamPersona;

// ==============================================================================
// Antigravity Skills Sub-Tab & Marketplace Controller
// ==============================================================================
function switchSkillsSubTab(subTab) {
  const tabs = ['active', 'marketplace', 'builder'];
  tabs.forEach(t => {
    const btn = document.getElementById(`skills-subtab-${t}`);
    const container = document.getElementById(`skills-container-${t}`);
    if (btn) {
      if (t === subTab) {
        btn.classList.add('bg-purple-600/30', 'border-purple-500/40', 'text-purple-200');
        btn.classList.remove('bg-white/5', 'text-zinc-400');
      } else {
        btn.classList.remove('bg-purple-600/30', 'border-purple-500/40', 'text-purple-200');
        btn.classList.add('bg-white/5', 'text-zinc-400');
      }
    }
    if (container) {
      if (t === subTab) {
        container.classList.remove('hidden');
        container.classList.add('animate-fade-in');
      } else {
        container.classList.add('hidden');
        container.classList.remove('animate-fade-in');
      }
    }
  });

  if (subTab === 'marketplace') {
    loadSkillsMarketplace();
  }
}
window.switchSkillsSubTab = switchSkillsSubTab;

async function loadSkillsMarketplace() {
  const grid = document.getElementById('skills-marketplace-grid');
  if (!grid) return;

  try {
    const res = await fetch('/api/skills/marketplace');
    const data = await res.json();
    const items = data.marketplace || [];

    grid.innerHTML = items.map(item => `
      <div class="glass-panel rounded-2xl p-5 border border-white/5 hover:border-purple-500/40 transition flex flex-col justify-between group">
        <div>
          <div class="flex items-start justify-between gap-2 mb-2">
            <div class="flex items-center gap-2.5">
              <span class="p-2 rounded-xl bg-purple-500/20 text-purple-300 text-lg">${escapeHtml(item.icon || '📦')}</span>
              <div>
                <h4 class="text-sm font-bold text-white group-hover:text-purple-300 transition">${escapeHtml(item.name)}</h4>
                <div class="text-[10px] text-zinc-400 font-mono">${escapeHtml(item.category || 'Agent Skill')} • v${escapeHtml(item.version || '1.0')}</div>
              </div>
            </div>
            <span class="text-[9px] px-2 py-0.5 rounded-full ${item.installed ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-purple-500/20 text-purple-300 border border-purple-500/30'} font-mono font-bold">
              ${item.installed ? 'INSTALLED' : 'ANTIGRAVITY'}
            </span>
          </div>

          <p class="text-xs text-zinc-300 leading-relaxed my-3">${escapeHtml(item.desc)}</p>

          <div class="p-2.5 rounded-xl bg-black/40 border border-white/5 space-y-1 my-2">
            <div class="text-[10px] font-mono text-zinc-400">Agent Instruction & Tools:</div>
            <div class="text-[11px] font-mono text-purple-200/90 truncate">${escapeHtml(item.system_prompt || '')}</div>
            <div class="flex flex-wrap gap-1 pt-1">
              ${(item.tools || []).map(t => `<span class="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-zinc-400 font-mono">${escapeHtml(t)}</span>`).join('')}
            </div>
          </div>
        </div>

        <div class="pt-3 border-t border-white/5 flex items-center justify-between gap-2">
          <span class="text-[10px] font-mono text-zinc-500">Autonomous Execution Ready</span>
          <button onclick="installSkillFromMarketplace('${item.id}', this)" ${item.installed ? 'disabled' : ''} class="px-4 py-2 rounded-xl ${item.installed ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-600/30'} text-xs font-bold transition flex items-center gap-1.5">
            <span>${item.installed ? '✓ Installed' : '+ Install Skill'}</span>
          </button>
        </div>
      </div>
    `).join('');
  } catch (err) {
    console.error("[Marketplace] Failed to fetch marketplace:", err);
  }
}
window.loadSkillsMarketplace = loadSkillsMarketplace;

async function installSkillFromMarketplace(skillId, btn) {
  if (btn) btn.disabled = true;
  try {
    const res = await fetch('/api/skills/install', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skill_id: skillId })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Install failed");
    alert(`Successfully installed "${data.skill.name}" into your autonomous studio!`);
    await refreshSkillsList();
    await loadSkillsMarketplace();
  } catch (err) {
    alert(`Install failed: ${err.message}`);
    if (btn) btn.disabled = false;
  }
}
window.installSkillFromMarketplace = installSkillFromMarketplace;

async function deployCustomAgent() {
  const name = document.getElementById('builder-agent-name')?.value.trim();
  const role = document.getElementById('builder-agent-role')?.value.trim();
  const desc = document.getElementById('builder-agent-desc')?.value.trim();
  const systemPrompt = document.getElementById('builder-agent-prompt')?.value.trim();
  const voice = document.getElementById('builder-agent-voice')?.value;

  const toolCheckboxes = document.querySelectorAll('.builder-tool-check:checked');
  const tools = Array.from(toolCheckboxes).map(c => c.value);

  if (!name || !desc || !systemPrompt) {
    alert("Please fill in Agent Name, Description, and System Prompt.");
    return;
  }

  try {
    const res = await fetch('/api/skills/custom', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        role: role,
        description: desc,
        system_prompt: systemPrompt,
        voice_archetype: voice,
        tools: tools
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Deploy failed");

    alert(`Autonomous Subagent "${name}" deployed successfully via Antigravity Agent Runtime!`);
    await refreshSkillsList();
    switchSkillsSubTab('active');
  } catch (err) {
    alert(`Deploy failed: ${err.message}`);
  }
}
window.deployCustomAgent = deployCustomAgent;

// ==============================================================================
// Multi-Model LLM Matrix Controller
// ==============================================================================
async function loadLLMModelsMatrix() {
  const container = document.getElementById('llm-models-matrix-grid');
  if (!container) return;

  try {
    const res = await fetch('/api/llm/models');
    const data = await res.json();
    state.models = data.models || [];

    container.innerHTML = state.models.map(m => `
      <div class="glass-panel rounded-2xl p-5 border border-white/5 hover:border-emerald-500/30 transition flex flex-col justify-between group">
        <div>
          <div class="flex items-start justify-between gap-2 mb-3">
            <div>
              <div class="text-xs font-bold text-white group-hover:text-emerald-300 transition flex items-center gap-1.5">
                <span>🧠</span> ${escapeHtml(m.name)}
              </div>
              <div class="text-[10px] text-zinc-400 font-mono">${escapeHtml(m.provider)}</div>
            </div>
            <span class="text-[9px] px-2 py-0.5 rounded-full ${m.ready ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-zinc-800 text-zinc-400'} font-mono font-bold uppercase">
              ${m.ready ? 'READY' : 'SETUP'}
            </span>
          </div>
          
          <div class="grid grid-cols-2 gap-2 my-3 p-2.5 rounded-xl bg-black/40 border border-white/5 text-[10px] font-mono text-zinc-400">
            <div>Context: <span class="text-white">${m.context_window || '128k'}</span></div>
            <div>Speed: <span class="text-emerald-400">${m.speed_tokens_sec || 'Fast'}</span></div>
          </div>
          <p class="text-xs text-zinc-300 leading-relaxed mb-3">${escapeHtml(m.desc || '')}</p>
        </div>

        <div class="pt-3 border-t border-white/5 flex items-center justify-between gap-2">
          <button onclick="selectLLMModel('${m.id}')" class="flex-1 py-1.5 rounded-xl bg-white/5 hover:bg-emerald-600/30 text-[11px] font-semibold text-zinc-300 hover:text-white transition text-center">
            Select Active
          </button>
          <button onclick="document.getElementById('llm-play-model').value = '${m.id}'; document.getElementById('llm-play-prompt').focus();" class="px-3 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-[11px] font-semibold text-white transition shadow-sm">
            Test
          </button>
        </div>
      </div>
    `).join('');
  } catch (err) {
    console.error("[LLM Matrix] Failed to load models:", err);
  }
}
window.loadLLMModelsMatrix = loadLLMModelsMatrix;

function selectLLMModel(modelId) {
  state.selectedModel = modelId;
  const headerSelect = document.getElementById('header-llm-model-select');
  if (headerSelect) headerSelect.value = modelId;
  const playSelect = document.getElementById('llm-play-model');
  if (playSelect) playSelect.value = modelId;
  alert(`Active reasoning model switched to: ${modelId}`);
}
window.selectLLMModel = selectLLMModel;

async function runLLMPlayground() {
  const prompt = document.getElementById('llm-play-prompt')?.value.trim();
  const modelId = document.getElementById('llm-play-model')?.value || 'local_neural';
  const resContainer = document.getElementById('llm-play-result');
  const btn = document.getElementById('btn-llm-generate-speak');

  if (!prompt || !resContainer) return;
  resContainer.classList.remove('hidden');
  resContainer.innerHTML = `
    <div class="text-xs text-indigo-300 flex items-center gap-2 py-2">
      <svg class="animate-spin w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg>
      Inferring on ${modelId.toUpperCase()} and synthesizing 24kHz audio...
    </div>
  `;
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/llm/chat-and-speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt,
        model_id: modelId,
        language: state.selectedLanguage || 'en',
        audio_prompt_url: state.referenceAudioUrl
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Multimodal generation failed");

    resContainer.innerHTML = `
      <div class="space-y-2.5">
        <div class="flex items-center justify-between text-xs border-b border-white/10 pb-2">
          <span class="font-bold text-white flex items-center gap-1.5">
            <span class="text-emerald-400">●</span> ${data.provider} (${data.model})
          </span>
          <span class="text-[10px] font-mono text-zinc-400">Duration: ${data.duration || '~4s'}</span>
        </div>
        <p class="text-xs text-zinc-200 leading-relaxed">${escapeHtml(data.text)}</p>
        ${data.audio_url ? `
          <div class="pt-2 border-t border-white/10 flex items-center gap-3">
            <audio controls src="${data.audio_url}" class="w-full h-8"></audio>
            <a href="${data.audio_url}" download="speech_${modelId}.wav" class="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex-shrink-0 transition">
              Download WAV
            </a>
          </div>
        ` : ''}
      </div>
    `;
  } catch (err) {
    resContainer.innerHTML = `<div class="text-xs text-rose-400">Error: ${err.message}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}
window.runLLMPlayground = runLLMPlayground;

// ==============================================================================
// Connectors & Integrations Controller
// ==============================================================================
async function loadConnectorsList() {
  try {
    const res = await fetch('/api/connectors/list');
    const data = await res.json();
    state.connectors = data.connectors || [];

    // Update badges
    state.connectors.forEach(c => {
      if (c.id === 'gemini') {
        const b = document.getElementById('badge-key-gemini');
        if (b) b.textContent = c.status.toUpperCase();
      }
      if (c.id === 'openai') {
        const b = document.getElementById('badge-key-openai');
        if (b) b.textContent = c.status.toUpperCase();
      }
      if (c.id === 'anthropic') {
        const b = document.getElementById('badge-key-anthropic');
        if (b) b.textContent = c.status.toUpperCase();
      }
    });
  } catch (err) {
    console.error("[Connectors] Failed to load connectors:", err);
  }
}
window.loadConnectorsList = loadConnectorsList;

async function saveApiKey(providerKey, inputId) {
  const val = document.getElementById(inputId)?.value.trim();
  if (!val) {
    alert("Please enter a valid API key string.");
    return;
  }

  try {
    const res = await fetch('/api/connectors/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider_key: providerKey, key_value: val })
    });
    const data = await res.json();
    if (res.ok) {
      alert(`Successfully saved ${providerKey}!`);
      loadConnectorsList();
      loadLLMModelsMatrix();
    } else {
      alert(`Error: ${data.detail || 'Failed to save'}`);
    }
  } catch (err) {
    alert(`Network Error: ${err.message}`);
  }
}
window.saveApiKey = saveApiKey;

async function triggerTestWebhook() {
  const rawPayload = document.getElementById('webhook-test-payload')?.value;
  const resContainer = document.getElementById('webhook-test-result');
  const btn = document.getElementById('btn-test-webhook');

  if (!resContainer) return;
  resContainer.classList.remove('hidden');
  resContainer.innerHTML = `<span class="text-zinc-400">Sending webhook payload...</span>`;
  if (btn) btn.disabled = true;

  try {
    const parsed = JSON.parse(rawPayload);
    const res = await fetch('/api/connectors/webhook', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(parsed)
    });
    const data = await res.json();

    if (res.ok && data.success) {
      resContainer.innerHTML = `
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-emerald-400 font-bold font-mono">● 200 OK • Inbound Event Accepted</span>
            <span class="text-zinc-400 font-mono text-[10px]">Duration: ${data.duration || '~3s'}</span>
          </div>
          <div class="text-zinc-300">${escapeHtml(data.text || '')}</div>
          ${data.audio_url ? `
            <audio controls src="${data.audio_url}" class="w-full h-8 mt-2"></audio>
          ` : ''}
        </div>
      `;
    } else {
      resContainer.innerHTML = `<span class="text-rose-400">Webhook Failed: ${data.detail || 'Error'}</span>`;
    }
  } catch (err) {
    resContainer.innerHTML = `<span class="text-rose-400">Invalid JSON or network failure: ${err.message}</span>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}
window.triggerTestWebhook = triggerTestWebhook;



