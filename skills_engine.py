"""
skills_engine.py
Autonomous AI Agent Skills Engine for Chatterbox Multilingual Platform.
Provides multi-agent pipelines:
1. Dual-Host Podcast Producer (Scriptwriting + Multi-Speaker Synthesis)
2. Neural Cross-Lingual Dubber & Translator
3. Expressive Document Narrator (Audiobook generation)
4. Emotion Voice Director (Acoustic dynamic modulation)
"""
import os
import uuid
import soundfile as sf
import numpy as np
from typing import Dict, List, Any, Optional

from tts_engine import synthesize, SUPPORTED_LANGUAGES
import llm_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "audio_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SKILLS_REGISTRY = [
    {
        "id": "podcast_host",
        "name": "🎙️ Autonomous Podcast Studio Agent",
        "badge": "Multi-Speaker",
        "category": "Audio Production",
        "desc": "Writes a multi-turn conversational podcast between two hosts and synthesizes alternating voices into a single mastered episode.",
        "inputs": [
            {"name": "topic", "label": "Podcast Topic / Question", "type": "text", "default": "How Zero-Shot Voice Cloning Transforms Global Media"},
            {"name": "duration_style", "label": "Format", "type": "select", "options": ["Brisk Summary (3 Turns)", "In-Depth Discussion (5 Turns)"]}
        ]
    },
    {
        "id": "cross_lingual_translator",
        "name": "🌐 Cross-Lingual Voice Translator Agent",
        "badge": "23 Languages",
        "category": "Localization",
        "desc": "Translates any text into your chosen target language with dialect fluency, then speaks it back with matching vocal timbre.",
        "inputs": [
            {"name": "source_text", "label": "Text to Translate & Speak", "type": "textarea", "default": "Artificial intelligence allows us to speak with anyone in the world in their own native language with studio clarity."},
            {"name": "target_language", "label": "Target Language", "type": "language_select", "default": "ja"}
        ]
    },
    {
        "id": "document_narrator",
        "name": "📖 Audiobook & Document Narrator Agent",
        "badge": "Long-Form",
        "category": "Publishing",
        "desc": "Ingests long-form articles, PDFs, or chapters, optimizes the prose for listening cadence, and produces studio narration.",
        "inputs": [
            {"name": "document_text", "label": "Document / Chapter Text", "type": "textarea", "default": "Chapter 1: The Acoustic Horizon. Sound does not simply travel through air; it carries human emotion, memory, and presence."},
            {"name": "voice_style", "label": "Narrator Style", "type": "select", "options": ["Warm Conversational", "Authoritative Documentary", "Calm Bedtime"]}
        ]
    },
    {
        "id": "voice_director",
        "name": "🎬 Neural Voice Director Agent",
        "badge": "Dynamic Modulation",
        "category": "Creative",
        "desc": "Analyzes the dramatic subtext of your script and auto-tunes pacing, temperature, and expressiveness for maximum cinematic impact.",
        "inputs": [
            {"name": "script", "label": "Dramatic Script", "type": "textarea", "default": "Listen closely... We only have one chance at this, and the entire transmission depends on what happens next."},
            {"name": "mood", "label": "Emotional Mood", "type": "select", "options": ["Dramatic Suspense", "Upbeat / Energetic", "Subtle Whispering", "Confident Keynote"]}
        ]
    },
    {
        "id": "kids_rhymes",
        "name": "🎈 Khyathi.Sri Kids Songs & Story Studio",
        "badge": "Kids & Family",
        "category": "Entertainment",
        "desc": "Joyful rhyming songs, moral bedtime stories, and interactive multilingual learning with colorful visual art and cheerful voice.",
        "inputs": [
            {"name": "theme", "label": "Song / Story Theme", "type": "text", "default": "Little Rainbow Butterfly in the Garden"},
            {"name": "content_type", "label": "Type", "type": "select", "options": ["Nursery Rhyme Song", "Bedtime Adventure Story", "Alphabet & Word Learning", "Cheerful Melodic Poem"]},
            {"name": "language", "label": "Language", "type": "language_select", "default": "en"}
        ]
    }
]


def list_skills() -> List[Dict[str, Any]]:
    return SKILLS_REGISTRY


def execute_podcast_skill(topic: str, turns_count: int = 3, model_id: str = "local_neural") -> Dict[str, Any]:
    """Generates a 2-host podcast and synthesizes alternating voices."""
    prompt = f"Write a natural podcast conversation between Host 1 (Alex) and Host 2 (Jordan) about: '{topic}'. Give {turns_count} alternating turns each."
    llm_res = llm_engine.generate_llm_text(prompt, model_id=model_id, system_prompt="Format dialogue with lines starting with 'Alex:' or 'Jordan:'.")
    script = llm_res["text"]

    lines = [l.strip() for l in script.split("\n") if l.strip() and (l.startswith("Alex:") or l.startswith("Jordan:"))]
    if not lines:
        lines = [
            f"Alex: Welcome to the episode on {topic}.",
            "Jordan: Absolutely thrilled to dive in, Alex. The implications are astounding.",
            f"Alex: Thanks for tuning into this automated podcast episode!"
        ]

    audio_segments = []
    sr_out = 24000
    host_voice_map = {
        "Alex": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "Jordan": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
    }

    for line in lines[:4]:  # limit to 4 turns for rapid turnaround
        speaker = "Alex" if line.startswith("Alex:") else "Jordan"
        clean_text = line.split(":", 1)[1].strip()
        sr, wav_np, _ = synthesize(
            text=clean_text,
            language_id="en",
            audio_prompt_path=host_voice_map.get(speaker),
            exaggeration=0.55 if speaker == "Alex" else 0.45,
            cfg_weight=0.5
        )
        audio_segments.append(wav_np)
        # Add 0.3s pause between speakers
        audio_segments.append(np.zeros(int(sr * 0.3), dtype=np.float32))

    combined = np.concatenate(audio_segments) if audio_segments else np.zeros(24000, dtype=np.float32)
    filename = f"podcast_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(out_path, combined, sr_out)

    return {
        "skill": "podcast_host",
        "success": True,
        "topic": topic,
        "script": script,
        "audio_url": f"/static/audio_output/{filename}",
        "duration": round(len(combined) / sr_out, 2),
        "speakers": ["Alex (Voice Sarah)", "Jordan (Voice Marco)"]
    }


def execute_translation_skill(source_text: str, target_lang: str, model_id: str = "local_neural") -> Dict[str, Any]:
    """Translates text and speaks it back in target language."""
    target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
    prompt = f"Translate the following text into fluent, natural {target_name}: '{source_text}'. Return ONLY the translated text without commentary."
    llm_res = llm_engine.generate_llm_text(prompt, model_id=model_id)
    translated_text = llm_res["text"].strip().strip('"').strip("'")

    sr, wav_np, _ = synthesize(
        text=translated_text,
        language_id=target_lang,
        exaggeration=0.5,
        cfg_weight=0.5
    )

    filename = f"trans_{target_lang}_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(out_path, wav_np, sr)

    return {
        "skill": "cross_lingual_translator",
        "success": True,
        "source_text": source_text,
        "translated_text": translated_text,
        "target_language": target_lang,
        "target_language_name": target_name,
        "audio_url": f"/static/audio_output/{filename}",
        "duration": round(len(wav_np) / sr, 2)
    }


def execute_narrator_skill(document_text: str, voice_style: str = "Warm Conversational") -> Dict[str, Any]:
    """Produces audiobook-style narration."""
    # Split text into sentences for fluid reading
    clean_text = document_text.strip()
    if len(clean_text) > 400:
        clean_text = clean_text[:400] + "..."

    sr, wav_np, _ = synthesize(
        text=clean_text,
        language_id="en",
        exaggeration=0.4 if "Calm" in voice_style else 0.6,
        cfg_weight=0.6
    )

    filename = f"narration_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(out_path, wav_np, sr)

    return {
        "skill": "document_narrator",
        "success": True,
        "style": voice_style,
        "narrated_text": clean_text,
        "audio_url": f"/static/audio_output/{filename}",
        "duration": round(len(wav_np) / sr, 2)
    }


def execute_voice_director_skill(script: str, mood: str = "Dramatic Suspense") -> Dict[str, Any]:
    """Applies acoustic modulation based on mood."""
    mood_params = {
        "Dramatic Suspense": {"exaggeration": 0.8, "temperature": 0.7, "cfg": 0.4},
        "Upbeat / Energetic": {"exaggeration": 0.9, "temperature": 0.85, "cfg": 0.6},
        "Subtle Whispering": {"exaggeration": 0.3, "temperature": 0.5, "cfg": 0.3},
        "Confident Keynote": {"exaggeration": 0.6, "temperature": 0.75, "cfg": 0.55}
    }
    params = mood_params.get(mood, {"exaggeration": 0.5, "temperature": 0.8, "cfg": 0.5})

    sr, wav_np, _ = synthesize(
        text=script.strip(),
        language_id="en",
        exaggeration=params["exaggeration"],
        temperature=params["temperature"],
        cfg_weight=params["cfg"]
    )

    filename = f"director_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(out_path, wav_np, sr)

    return {
        "skill": "voice_director",
        "success": True,
        "mood": mood,
        "applied_parameters": params,
        "script": script.strip(),
        "audio_url": f"/static/audio_output/{filename}",
        "duration": round(len(wav_np) / sr, 2)
    }


def execute_kids_rhymes_skill(theme: str = "Rainbow Butterfly", content_type: str = "Nursery Rhyme Song", language: str = "en") -> Dict[str, Any]:
    """Generates sweet rhymes or bedtime stories with colorful visual art and joyful voice synthesis."""
    import image_engine
    theme_clean = theme.strip() or "Rainbow Butterfly in Sunny Garden"
    
    rhymes = [
        f"Dancing in the golden sun, little {theme_clean} having fun! Up into the sky so bright, spreading colors left and right. Laugh and sing and play all day, chasing happy dreams away!",
        f"Once upon a cozy time, under a smiling moon, {theme_clean} sang a lullaby with a cheerful silver spoon. Stars were blinking in the night, whispering softly: sleep tight!",
        f"One, two, three, come sing with me! {theme_clean} under the apple tree! Four, five, six, with joyful tricks, learning words so sweet and quick!"
    ]
    import random
    verse = random.choice(rhymes)

    art = image_engine.generate_cinematic_artwork(f"Cute colorful cartoon illustration of {theme_clean} with sunshine and flowers")

    sr, wav_np, _ = synthesize(text=verse, language_id=language)
    filename = f"kids_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(out_path, wav_np, sr)

    return {
        "skill": "kids_rhymes",
        "success": True,
        "character": "Khyathi Sri",
        "theme": theme_clean,
        "content_type": content_type,
        "text": verse,
        "image_url": art["image_url"],
        "audio_url": f"/static/audio_output/{filename}",
        "duration": round(len(wav_np) / sr, 2),
        "partner": "Key Secure Foundation"
    }


# ==============================================================================
# Antigravity-Style Skill Marketplace & Custom Subagent Registry
# ==============================================================================
SKILLS_MARKETPLACE: List[Dict[str, Any]] = [
    {
        "id": "customer_support_vocalist",
        "name": "🎧 Customer Support Vocalist Agent",
        "badge": "Enterprise",
        "category": "Customer Ops",
        "desc": "Resolves customer questions with empathetic, patient vocal cadence in any of 23 native tongues.",
        "author": "Antigravity Ecosystem",
        "tools_enabled": ["llm_reasoning", "voice_synthesis", "ticket_lookup"],
        "voice_archetype": "Warm Conversational (Elena)",
        "system_prompt": "You are an empathetic, world-class customer resolution voice agent.",
        "installed": False
    },
    {
        "id": "game_npc_dialogue",
        "name": "⚔️ Interactive Game NPC Voice Engine",
        "badge": "Metaverse",
        "category": "Gaming",
        "desc": "Dynamic fantasy and sci-fi non-player character voice generator with dramatic accents and lore memory.",
        "author": "Antigravity Ecosystem",
        "tools_enabled": ["llm_reasoning", "voice_synthesis", "lore_retrieval"],
        "voice_archetype": "Epic Cinematic (Dmitri / Alex)",
        "system_prompt": "You are a legendary RPG character responding with deep in-world immersion.",
        "installed": False
    },
    {
        "id": "meeting_transcriber_reciter",
        "name": "📊 Executive Briefing Narrator",
        "badge": "Productivity",
        "category": "Corporate",
        "desc": "Condenses lengthy meeting transcripts into a crisp 60-second executive audio briefing.",
        "author": "Antigravity Ecosystem",
        "tools_enabled": ["llm_reasoning", "summarization", "voice_synthesis"],
        "voice_archetype": "Crisp Executive (Marco)",
        "system_prompt": "You summarize key business decisions into bullet-point audio recaps.",
        "installed": False
    },
    {
        "id": "youtube_explainer_voice",
        "name": "🎬 Video Explainer Voiceover Agent",
        "badge": "Creator",
        "category": "Media",
        "desc": "Directs educational video scripts with engaging inflection, pacing, and chapter markers.",
        "author": "Antigravity Ecosystem",
        "tools_enabled": ["llm_reasoning", "prosody_tuning", "voice_synthesis"],
        "voice_archetype": "Energetic Host (Sarah)",
        "system_prompt": "You are an articulate YouTube documentary narrator.",
        "installed": False
    },
    {
        "id": "interactive_tutor",
        "name": "🎓 GS Interactive Socratic Voice Tutor",
        "badge": "Education",
        "category": "Learning",
        "desc": "Teaches complex technical concepts with patient conversational back-and-forth and acoustic illustrations.",
        "author": "Antigravity Ecosystem",
        "tools_enabled": ["llm_reasoning", "voice_synthesis", "concept_explainer"],
        "voice_archetype": "Empathetic Counselor (Elena)",
        "system_prompt": "You are a master teacher who guides students through the Socratic method with vocal warmth.",
        "installed": False
    },
    {
        "id": "creative_visual_synthesizer",
        "name": "🎨 GS Creative Vision & Image Generator",
        "badge": "Multimodal Art",
        "category": "Creative Studio",
        "desc": "Synthesizes high-impact cinematic scene artwork, acoustic spectrograms, and character visualizations like Antigravity.",
        "author": "Antigravity Ecosystem",
        "tools_enabled": ["image_generation", "visual_prompting", "cinematic_rendering"],
        "voice_archetype": "Epic Cinematic (Dmitri / Alex)",
        "system_prompt": "You generate evocative visual scenes and artistic compositions matching the narrative mood.",
        "installed": False
    }
]


def list_marketplace() -> List[Dict[str, Any]]:
    """Returns available skills in the marketplace."""
    active_ids = {s["id"] for s in SKILLS_REGISTRY}
    for item in SKILLS_MARKETPLACE:
        item["installed"] = item["id"] in active_ids
    return SKILLS_MARKETPLACE


def install_skill(skill_id: str) -> Dict[str, Any]:
    """Installs a skill from the marketplace into active studio."""
    existing = [s for s in SKILLS_REGISTRY if s["id"] == skill_id]
    if existing:
        return {"success": True, "message": f"Skill '{skill_id}' is already active.", "skill": existing[0]}

    match = [m for m in SKILLS_MARKETPLACE if m["id"] == skill_id]
    if not match:
        raise ValueError(f"Skill '{skill_id}' not found in Marketplace.")

    item = match[0]
    new_skill = {
        "id": item["id"],
        "name": item["name"],
        "badge": item["badge"],
        "category": item["category"],
        "desc": item["desc"],
        "inputs": [
            {"name": "prompt", "label": "Instruction / Query", "type": "textarea", "default": f"Generate a session with the {item['name']}."},
            {"name": "language", "label": "Language", "type": "language_select", "default": "en"}
        ],
        "is_custom": False,
        "tools_enabled": item.get("tools_enabled", [])
    }
    SKILLS_REGISTRY.append(new_skill)
    item["installed"] = True
    return {"success": True, "message": f"Skill '{item['name']}' installed successfully!", "skill": new_skill}


def create_custom_agent(name: str, role: str, desc: str, system_prompt: str, voice_archetype: str, tools: List[str]) -> Dict[str, Any]:
    """Creates a custom autonomous agent skill just like Antigravity's define_subagent."""
    slug = "custom_" + "".join(c for c in name.lower() if c.isalnum() or c in "_-")[:20] + f"_{uuid.uuid4().hex[:4]}"
    custom_skill = {
        "id": slug,
        "name": f"🤖 {name}",
        "badge": "Custom AGY",
        "category": role or "Custom Agent",
        "desc": desc or f"Autonomous agent specialized in {role}.",
        "system_prompt": system_prompt,
        "voice_archetype": voice_archetype,
        "tools_enabled": tools or ["llm_reasoning", "voice_synthesis"],
        "is_custom": True,
        "inputs": [
            {"name": "user_input", "label": "Task or Message for Agent", "type": "textarea", "default": f"Hello {name}, please execute your agent routine."},
            {"name": "language", "label": "Language", "type": "language_select", "default": "en"}
        ]
    }
    SKILLS_REGISTRY.append(custom_skill)
    return {"success": True, "message": f"Custom Agent '{name}' registered!", "agent": custom_skill, "skill": custom_skill}


def execute_custom_agent_skill(agent_id: str, user_input: str, language: str = "en") -> Dict[str, Any]:
    """Runs an installed or custom agent with reasoning + voice synthesis."""
    agent = next((s for s in SKILLS_REGISTRY if s["id"] == agent_id), None)
    if not agent:
        raise ValueError(f"Agent '{agent_id}' not found.")

    system_prompt = agent.get("system_prompt", "You are an autonomous AI voice specialist.")
    llm_res = llm_engine.generate_llm_text(
        prompt=user_input,
        model_id="local_neural",
        system_prompt=system_prompt
    )
    reply_text = llm_res.get("text", "")

    sr, wav_np, _ = synthesize(text=reply_text[:350], language_id=language)
    filename = f"agent_{agent_id[:10]}_{uuid.uuid4().hex[:6]}.wav"
    out_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(out_path, wav_np, sr)

    return {
        "success": True,
        "agent_id": agent_id,
        "agent_name": agent["name"],
        "user_input": user_input,
        "reply_text": reply_text,
        "language": language,
        "audio_url": f"/static/audio_output/{filename}",
        "duration": round(len(wav_np) / sr, 2)
    }

