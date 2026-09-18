"""
connectors.py
Connectors, Integrations & Webhook Dispatcher for Chatterbox Multilingual Platform.
Enables connections with:
1. LLM API Providers (Gemini, OpenAI, DeepSeek, Groq, Ollama)
2. Webhooks (Inbound synthesis triggers & Outbound event callbacks)
3. Cloud Storage (S3 / GCS export hooks)
4. YouTube & External Media Audio Extraction
"""
import os
import json
import time
from typing import Dict, List, Any, Optional
import llm_engine

CONNECTORS_CATALOG = [
    {
        "id": "gemini_connector",
        "name": "GS Multilingual Multimodal Gateway",
        "category": "Cognitive Cloud",
        "status": "active" if llm_engine.API_KEYS.get("GEMINI_API_KEY") else "needs_key",
        "desc": "Ultra low-latency multimodal reasoning pipeline for live speech transcription and 23-language translation.",
        "config_fields": ["GEMINI_API_KEY"]
    },
    {
        "id": "openai_connector",
        "name": "GS Omni Intelligence Gateway",
        "category": "Cognitive Cloud",
        "status": "active" if llm_engine.API_KEYS.get("OPENAI_API_KEY") else "needs_key",
        "desc": "Omni-modal conversational intelligence for expressive voice storytelling, podcast scripting, and prompt shaping.",
        "config_fields": ["OPENAI_API_KEY"]
    },
    {
        "id": "deepseek_connector",
        "name": "GS Frontier MoE Cognitive Gateway",
        "category": "Cognitive Cloud",
        "status": "active" if llm_engine.API_KEYS.get("DEEPSEEK_API_KEY") else "needs_key",
        "desc": "671B Mixture-of-Experts high-parameter engine for complex multilingual phonetics and literary chapter analysis.",
        "config_fields": ["DEEPSEEK_API_KEY"]
    },
    {
        "id": "webhook_connector",
        "name": "Inbound Webhook API / Zapier / Make",
        "category": "Automation",
        "status": "ready",
        "desc": "POST endpoint (/api/connectors/webhook) to trigger automated speech synthesis from external automations, CRMs, or bots.",
        "endpoint": "/api/connectors/webhook"
    },
    {
        "id": "cloud_storage_connector",
        "name": "Cloud Storage & S3 Exporter",
        "category": "Storage",
        "status": "ready",
        "desc": "Enables automated sync of rendered 24kHz master audio files to cloud object storage buckets.",
        "config_fields": ["S3_BUCKET_NAME", "AWS_ACCESS_KEY_ID"]
    },
    {
        "id": "youtube_media_connector",
        "name": "Media Ingestion & Audio Extractor",
        "category": "Media",
        "status": "ready",
        "desc": "Pulls reference audio clips from YouTube or direct media URLs for instant zero-shot voice cloning.",
        "config_fields": ["MAX_DURATION_SECONDS"]
    }
]


def list_connectors() -> List[Dict[str, Any]]:
    # Update live status based on keys
    for c in CONNECTORS_CATALOG:
        if c["id"] == "gemini_connector":
            c["status"] = "connected" if llm_engine.API_KEYS.get("GEMINI_API_KEY") else "needs_key"
        elif c["id"] == "openai_connector":
            c["status"] = "connected" if llm_engine.API_KEYS.get("OPENAI_API_KEY") else "needs_key"
        elif c["id"] == "anthropic_connector":
            c["status"] = "connected" if llm_engine.API_KEYS.get("ANTHROPIC_API_KEY") else "needs_key"
    return CONNECTORS_CATALOG


def handle_inbound_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Processes an inbound webhook trigger for synthesis or agent execution."""
    text = payload.get("text", "")
    language = payload.get("language", "en")
    action = payload.get("action", "synthesize")

    if not text.strip():
        raise ValueError("Webhook payload must include non-empty 'text'.")

    from tts_engine import synthesize
    sr, wav_np, _ = synthesize(text=text.strip(), language_id=language)

    import uuid
    import soundfile as sf
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "static", "audio_output")
    filename = f"webhook_{uuid.uuid4().hex[:8]}.wav"
    sf.write(os.path.join(output_dir, filename), wav_np, sr)

    return {
        "success": True,
        "action": action,
        "audio_url": f"/static/audio_output/{filename}",
        "duration": round(len(wav_np) / sr, 2),
        "text": text,
        "language": language
    }
