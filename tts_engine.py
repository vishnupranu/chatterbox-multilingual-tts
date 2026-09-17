"""
tts_engine.py
Shared TTS Engine for Chatterbox Multilingual TTS.
Supports model loading (V3/V2), MPS/CUDA/CPU device management, and audio synthesis.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
from typing import Optional, Tuple
import numpy as np
import torch
import torchaudio as ta

from mac_patch import apply_device_patch, get_target_device

# Supported languages mapping (ISO code -> Full Name)
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

# Default sample prompts and reference audio for languages
SAMPLE_CONFIG = {
    "en": {
        "text": "Welcome to Chatterbox Multilingual TTS. This voice is generated locally with high fidelity.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
    },
    "fr": {
        "text": "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
    },
    "es": {
        "text": "Hola, bienvenido al sintetizador de voz multilingüe Chatterbox de última generación.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
    },
    "de": {
        "text": "Guten Tag, dies ist das mehrsprachige Chatterbox Sprachmodell.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
    },
    "it": {
        "text": "Buongiorno, questo è il sistema di sintesi vocale multilingue Chatterbox.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
    },
    "pt": {
        "text": "Olá, bem-vindo ao modelo de síntese de voz multilíngue Chatterbox.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
    },
    "hi": {
        "text": "नमस्ते, यह चैटरबॉक्स बहुभाषी टेक्स्ट-टू-स्पीच प्रणाली है।",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
    },
    "zh": {
        "text": "你好，这是 Chatterbox 多语言语音合成系统，能够生成自然流畅的语音。",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
    },
    "ja": {
        "text": "こんにちは、Chatterbox 多言語音声合成システムへようこそ。",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
    },
    "ko": {
        "text": "안녕하세요, 채터박스 다국어 음성 합성 시스템에 오신 것을 환영합니다.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
    },
    "ar": {
        "text": "مرحباً بكم في نظام شاتربوكس متعدد اللغات لتحويل النص إلى كلام.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
    },
    "ru": {
        "text": "Здравствуйте, это многоязычная модель синтеза речи Chatterbox.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
    },
    "nl": {
        "text": "Hallo, dit is de meertalige tekst-naar-spraak synthese van Chatterbox.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
    },
    "pl": {
        "text": "Witaj w wielojęzycznym systemie syntezy mowy Chatterbox.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
    },
    "tr": {
        "text": "Merhaba, Chatterbox çok dilli metinden sese dönüştürme sistemine hoş geldiniz.",
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
    },
}

_MODEL_INSTANCE = None
_CURRENT_T3 = None
_CURRENT_DEV = None


def set_random_seed(seed: int):
    """Set random seed across torch, numpy, and random."""
    if seed != 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)


def get_model(device: Optional[str] = None, t3_model: str = "v3"):
    """
    Loads or returns the singleton instance of ChatterboxMultilingualTTS.
    """
    global _MODEL_INSTANCE, _CURRENT_T3, _CURRENT_DEV

    target_device = device or get_target_device()
    apply_device_patch(target_device)

    if _MODEL_INSTANCE is None or _CURRENT_T3 != t3_model or _CURRENT_DEV != target_device:
        print(f"Loading ChatterboxMultilingualTTS (t3_model='{t3_model}', device='{target_device}')...")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        _MODEL_INSTANCE = ChatterboxMultilingualTTS.from_pretrained(
            device=target_device,
            t3_model=t3_model
        )
        if hasattr(_MODEL_INSTANCE, "to") and str(getattr(_MODEL_INSTANCE, "device", "")) != target_device:
            _MODEL_INSTANCE.to(target_device)

        _CURRENT_T3 = t3_model
        _CURRENT_DEV = target_device
        print(f"Model successfully loaded on {target_device}!")

    return _MODEL_INSTANCE


def resolve_audio_prompt(audio_prompt: Optional[str]) -> Optional[str]:
    """
    If audio_prompt is a URL, download it to local cache directory.
    If it's a local path, ensure it exists.
    If None, returns None to use built-in voice conditionals.
    """
    if not audio_prompt or not str(audio_prompt).strip():
        return None

    audio_prompt = str(audio_prompt).strip()

    # If it's a URL, download and cache it locally
    if audio_prompt.startswith("http://") or audio_prompt.startswith("https://"):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "prompts")
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(audio_prompt.split("?")[0])
        local_path = os.path.join(cache_dir, filename)

        if not os.path.exists(local_path):
            import requests
            print(f"Downloading reference audio prompt from {audio_prompt}...")
            resp = requests.get(audio_prompt, timeout=30)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            print(f"Downloaded to {local_path}")
        return local_path

    if os.path.exists(audio_prompt):
        return audio_prompt

    print(f"Warning: Audio prompt file '{audio_prompt}' not found. Using default voice.")
    return None


def synthesize(
    text: str,
    language_id: str,
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    seed: int = 0,
    device: Optional[str] = None,
    t3_model: str = "v3",
) -> Tuple[int, np.ndarray, torch.Tensor]:
    """
    Synthesizes speech from text.
    Returns (sample_rate, numpy_waveform, torch_waveform_tensor).
    """
    if language_id not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{language_id}'. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    set_random_seed(seed)
    model = get_model(device=device, t3_model=t3_model)

    gen_kwargs = {
        "exaggeration": float(exaggeration),
        "temperature": float(temperature),
        "cfg_weight": float(cfg_weight),
    }

    # Resolve audio prompt (downloads URL if needed, or falls back to built-in voice)
    ref_audio = resolve_audio_prompt(audio_prompt_path)
    if ref_audio:
        gen_kwargs["audio_prompt_path"] = ref_audio
        print(f"Using reference audio: {ref_audio}")
    else:
        print("Using built-in default voice.")

    # Generate audio
    wav_tensor = model.generate(
        text=text[:300],  # Maximum 300 characters per inference chunk
        language_id=language_id,
        **gen_kwargs
    )

    sr = model.sr
    wav_np = wav_tensor.squeeze().cpu().numpy()
    return sr, wav_np, wav_tensor


def save_audio(output_path: str, wav_tensor: torch.Tensor, sample_rate: int):
    """Saves generated waveform tensor to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    ta.save(output_path, wav_tensor.cpu(), sample_rate)
