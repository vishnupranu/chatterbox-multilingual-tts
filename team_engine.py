"""
team_engine.py
Team and Character Agent Registry for GS Voice Intelligence Platform.
Defines human founder and AI specialist personas with voice auditioning and cloning integration.
"""
from typing import List, Dict, Any

TEAM_MEMBERS: List[Dict[str, Any]] = [
    {
        "id": "char_founder",
        "name": "G.S.",
        "title": "Lead AI Architect & Founder",
        "role": "Creator & Neural Audio Researcher",
        "tag": "CREATOR",
        "tag_color": "border-orange-500/40 text-orange-300 bg-orange-500/10",
        "avatar": "/static/img/avatar_founder_human.jpg",
        "preset_voice": "ru_male",
        "bio": "Pioneered the flow-matching neural architecture for 24kHz zero-shot speech. Spearheading multi-modal agentic intelligence and cross-lingual voice transfer.",
        "sample_quote": "We built Chatterbox to preserve authentic human emotional timbre across 23 languages without phonetic compromise.",
        "specialties": ["Flow Matching", "Zero-Shot Timbre", "Neural Audio DSP"],
        "status": "In Studio",
        "active": True
    },
    {
        "id": "char_elena",
        "name": "Elena Vance",
        "title": "Cognitive Linguistics Lead",
        "role": "Cross-Lingual Phonetics & Cultural Prosody",
        "tag": "LINGUISTICS",
        "tag_color": "border-cyan-500/40 text-cyan-300 bg-cyan-500/10",
        "avatar": "/static/img/avatar_elena.svg",
        "preset_voice": "fr_female",
        "bio": "Specializes in multi-accent prosody transfer. Ensures that cross-lingual voice dubbing captures subtle cultural nuance rather than mechanical translation.",
        "sample_quote": "Cross-lingual dubbing must breathe with native cadence. When the voice matches the culture, barriers vanish.",
        "specialties": ["23 Phonetic Alphabets", "Accent Neutralization", "Cultural Pacing"],
        "status": "Active Agent",
        "active": True
    },
    {
        "id": "char_kaelen",
        "name": "Kaelen Thorne",
        "title": "Autonomous Agent Systems Architect",
        "role": "Antigravity Agentic Tooling & Multi-LLM Orchestration",
        "tag": "AGENTS",
        "tag_color": "border-purple-500/40 text-purple-300 bg-purple-500/10",
        "avatar": "/static/img/avatar_kaelen.svg",
        "preset_voice": "it_male",
        "bio": "Architect of autonomous subagents chaining Google Gemini, OpenAI GPT-4o, DeepSeek-V3, LLaMA 3.3, and local neural weights into self-executing audio production pipelines.",
        "sample_quote": "Giving AI agents expressive 24kHz speech and autonomous tool access unlocks completely conversational computing.",
        "specialties": ["Antigravity Subagents", "Multi-LLM Routing", "Tool-Calling Pipelines"],
        "status": "Active Agent",
        "active": True
    },
    {
        "id": "char_marcus",
        "name": "Dr. Marcus Vance",
        "title": "Audio DSP & Vocoder Scientist",
        "role": "HiFi-GAN & Acoustic Phase Coherence",
        "tag": "DSP & ACOUSTICS",
        "tag_color": "border-emerald-500/40 text-emerald-300 bg-emerald-500/10",
        "avatar": "/static/img/avatar_marcus.svg",
        "preset_voice": "en_female",
        "bio": "Expert in non-autoregressive acoustic flow, 24kHz phase unwrapping, and hardware acceleration on Apple Silicon MPS and CUDA GPUs.",
        "sample_quote": "Studio fidelity means 24,000 samples per second of clean harmonic resolution with sub-100ms first-chunk generation.",
        "specialties": ["24kHz HiFi-GAN", "Apple Silicon MPS Tuning", "Phase Alignment"],
        "status": "Active Agent",
        "active": True
    },
    {
        "id": "char_khyathi",
        "name": "Khyathi Sri",
        "title": "Creative Entertainment & Storytelling Star",
        "role": "Kids Entertainment, Melodic Rhymes & Multilingual Learning",
        "tag": "ENTERTAINMENT & KIDS",
        "tag_color": "border-pink-500/40 text-pink-300 bg-pink-500/10",
        "avatar": "/static/img/avatar_khyathi_sri.png",
        "preset_voice": "en_female",
        "bio": "Beloved creative storytelling persona bringing joyful fairy tales, nursery rhymes, children's songs, and playful English and multilingual learning to families worldwide.",
        "sample_quote": "Welcome to the magical world of songs and fairy tales! Let's sing together, discover new words, and let our imagination fly across the rainbow!",
        "specialties": ["Kids Rhymes & Melodies", "Bedtime Fairy Tales", "Interactive Vocabulary", "Joyful Cadence"],
        "partner": "Key Secure Foundation",
        "status": "Star Performer",
        "active": True
    }
]


def get_team_roster() -> List[Dict[str, Any]]:
    """Returns all team personas."""
    return TEAM_MEMBERS


def get_member_by_id(member_id: str) -> Dict[str, Any]:
    for m in TEAM_MEMBERS:
        if m["id"] == member_id:
            return m
    return TEAM_MEMBERS[0]
