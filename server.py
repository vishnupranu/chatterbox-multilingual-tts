"""
server.py
Full-stack FastAPI server powering the Z.ai-style Chatterbox Multilingual platform.
Serves interactive Chat, Code workspace, autonomous Workers, and REST API.
"""
import io
import os
import uuid
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Header
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tts_engine import (
    SUPPORTED_LANGUAGES,
    SAMPLE_CONFIG,
    synthesize,
    get_target_device,
)
import workers
import billing
import copilot
import llm_engine
import skills_engine
import connectors
import team_engine
import image_engine
import auth

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(STATIC_DIR, "audio_output")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
IMG_OUTPUT_DIR = os.path.join(STATIC_DIR, "img_output")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="Chatterbox Z.ai Multilingual Platform",
    description="Full-stack AI voice synthesis platform modeled after chat.z.ai with Chat, Code, and Workers.",
    version="2.0.0",
)


# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatSynthesisRequest(BaseModel):
    text: str = Field(..., description="Message text to synthesize")
    language: str = Field("en", description="2-letter ISO language code")
    audio_prompt_url: Optional[str] = Field(None, description="Reference audio file path or URL for voice cloning")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    seed: int = Field(0)
    model_version: str = Field("v3")


class CodeRunRequest(BaseModel):
    code_type: str = Field("python", description="Language: python, curl, or js")
    text: str = Field("Hello from Chatterbox Z.ai Code Workspace!", description="Text to synthesize")
    language: str = Field("en", description="Target language")
    exaggeration: float = Field(0.5)
    cfg_weight: float = Field(0.5)


class BatchWorkerRequest(BaseModel):
    texts: List[str] = Field(..., description="List of sentence chunks to synthesize")
    language: str = Field("en", description="Target language")
    ref_audio: Optional[str] = Field(None)
    exaggeration: float = Field(0.5)
    cfg_weight: float = Field(0.5)


class DubbingWorkerRequest(BaseModel):
    source_text: str = Field(..., description="Source text to dub")
    target_languages: List[str] = Field(..., description="Target language codes")
    ref_audio: Optional[str] = Field(None)


@app.get("/")
def index():
    """Serves the primary Z.ai Single Page Application."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Chatterbox Z.ai Platform API is running. index.html not yet initialized."}


@app.get("/api/status")
def get_system_status():
    """Returns runtime engine status, device backend, and models."""
    return {
        "engine": "Chatterbox Multilingual TTS",
        "device": get_target_device(),
        "model_version": "v3",
        "supported_languages_count": len(SUPPORTED_LANGUAGES),
        "status": "ready",
        "features": ["zero-shot-cloning", "mps-acceleration", "24khz-hi-fi", "autonomous-workers"]
    }


# =====================================================================
# AUTHENTICATION & USER MANAGEMENT
# =====================================================================

class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password (min 6 chars)")
    name: str = Field("", description="User display name")
    role: str = Field("Student", description="User role: Student, Researcher, Professional, Educator")


class LoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class LogoutRequest(BaseModel):
    token: str = Field(..., description="Active session token")


@app.post("/api/auth/register")
def api_auth_register(req: RegisterRequest):
    """Registers a new user account with role assignment."""
    res = auth.register_user(
        email=req.email,
        password=req.password,
        name=req.name,
        role=req.role
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Registration failed."))
    return res


@app.post("/api/auth/login")
def api_auth_login(req: LoginRequest):
    """Authenticates credentials and returns a secure session token."""
    res = auth.login_user(email=req.email, password=req.password)
    if not res.get("success"):
        raise HTTPException(status_code=401, detail=res.get("error", "Invalid email or password."))
    return res


@app.get("/api/auth/me")
def api_auth_me(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    """Validates session token and returns the current user profile."""
    auth_token = token
    if not auth_token and authorization:
        if authorization.lower().startswith("bearer "):
            auth_token = authorization[7:].strip()
        else:
            auth_token = authorization.strip()

    if not auth_token:
        raise HTTPException(status_code=401, detail="Authentication token required.")

    user = auth.get_current_user(auth_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    return {"success": True, "user": user}


@app.post("/api/auth/logout")
def api_auth_logout(req: LogoutRequest):
    """Terminates an active session."""
    auth.logout_user(req.token)
    return {"success": True, "message": "Successfully logged out."}


# =====================================================================
# ADMIN & SUPER ADMIN PERMISSION ENDPOINTS
# =====================================================================

@app.get("/api/admin/users")
def api_admin_users(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    """Super Admin Only: Lists all registered platform users."""
    auth_token = token
    if not auth_token and authorization:
        auth_token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else authorization.strip()
    if not auth_token:
        raise HTTPException(status_code=401, detail="Authentication token required.")
    users = auth.list_all_users(auth_token)
    if users is None:
        raise HTTPException(status_code=403, detail="Forbidden: Super Admin permission mandated.")
    return {"success": True, "users": users}


@app.get("/api/admin/stats")
def api_admin_stats(token: Optional[str] = None, authorization: Optional[str] = Header(None)):
    """Admin & Super Admin: Returns platform health, security metrics, and sessions."""
    auth_token = token
    if not auth_token and authorization:
        auth_token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else authorization.strip()
    if not auth_token:
        raise HTTPException(status_code=401, detail="Authentication token required.")
    stats = auth.get_admin_system_stats(auth_token)
    if stats is None:
        raise HTTPException(status_code=403, detail="Forbidden: Admin permission mandated.")
    return {"success": True, "stats": stats}


class CreateOrderRequest(BaseModel):
    plan_id: str = Field(..., description="Plan ID: plan_starter, plan_pro, or plan_enterprise")


class VerifyOrderRequest(BaseModel):
    order_id: str = Field(..., description="Order ID to verify")


@app.get("/api/billing/balance")
def get_user_balance():
    """Returns the user's available credits."""
    return billing.get_balance()


@app.get("/api/billing/plans")
def get_billing_plans():
    """Returns all available credit packages and pricing."""
    return {"plans": billing.PLANS}


@app.post("/api/billing/create-order")
def create_billing_order(req: CreateOrderRequest):
    """Creates a Scan to Pay order with QR payload."""
    try:
        order = billing.create_scan_to_pay_order(req.plan_id)
        return {"success": True, "order": order}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/billing/verify-payment")
def verify_billing_payment(req: VerifyOrderRequest):
    """Verifies payment for an order and credits balance."""
    try:
        res = billing.verify_order_payment(req.order_id)
        return res
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class CopilotChatRequest(BaseModel):
    query: str = Field(..., description="User query for the project-trained AITalk copilot")
    context: Optional[str] = Field(None, description="Optional active workspace context")


@app.post("/api/copilot/chat")
def copilot_chat(req: CopilotChatRequest):
    """AITalk project-trained copilot endpoint answering queries about Chatterbox architecture, languages, code, and workers."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    res = copilot.ask_aitalk(req.query, req.context)
    return {"success": True, **res}


# ==============================================================================
# LLM Models, Agent Skills & Connectors Endpoints
# ==============================================================================
class LLMGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to send to selected LLM")
    model_id: str = Field("local_neural", description="Model ID")
    system_prompt: Optional[str] = Field(None)


class LLMChatAndSpeakRequest(BaseModel):
    prompt: str = Field(..., description="User prompt for conversational LLM response")
    model_id: str = Field("local_neural", description="LLM model: local_neural, gemini_flash, gemini_pro, gpt4o, deepseek_v3, llama_33, ollama_local")
    language: str = Field("en", description="Target language to synthesize response in")
    audio_prompt_url: Optional[str] = Field(None, description="Reference audio for voice cloning")


class ExecuteSkillRequest(BaseModel):
    skill_id: str = Field(..., description="Skill ID to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Skill execution parameters")


class SetApiKeyRequest(BaseModel):
    provider_key: str = Field(..., description="Provider environment variable name e.g. GEMINI_API_KEY")
    key_value: str = Field(..., description="API key value")


@app.get("/api/llm/models")
def get_llm_models():
    """Returns all supported LLM models and their active configuration state."""
    return {"models": llm_engine.get_available_models()}


@app.post("/api/llm/generate")
def generate_llm_completion(req: LLMGenerateRequest):
    """Executes prompt on selected LLM."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    return llm_engine.generate_llm_text(req.prompt, req.model_id, req.system_prompt)


@app.post("/api/llm/chat-and-speak")
def chat_and_speak(req: LLMChatAndSpeakRequest):
    """Full multimodal pipeline: LLM generates text reasoning, Chatterbox synthesizes it to 24kHz audio."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    # 1. Generate text from chosen LLM
    llm_res = llm_engine.generate_llm_text(req.prompt, req.model_id)
    text_reply = llm_res.get("text", "")
    
    # 2. Synthesize generated response
    lang = req.language.lower()
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"
    
    try:
        import soundfile as sf
        sr, wav_np, _ = synthesize(
            text=text_reply[:400],
            language_id=lang,
            audio_prompt_path=req.audio_prompt_url
        )
        audio_id = f"llm_voice_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(OUTPUT_DIR, audio_id)
        sf.write(output_path, wav_np, sr, format="WAV")

        return {
            "success": True,
            "prompt": req.prompt,
            "text": text_reply,
            "model": llm_res.get("model", req.model_id),
            "provider": llm_res.get("provider", "Local Neural"),
            "audio_url": f"/static/audio_output/{audio_id}",
            "duration": round(len(wav_np) / sr, 2),
            "language": lang
        }
    except Exception as e:
        return {
            "success": True,
            "prompt": req.prompt,
            "text": text_reply,
            "model": llm_res.get("model", req.model_id),
            "audio_url": None,
            "warning": f"Synthesis bypassed: {str(e)}"
        }


@app.get("/api/skills/list")
def list_agent_skills():
    """Returns all registered autonomous AI Agent Skills."""
    return {"skills": skills_engine.list_skills()}


@app.post("/api/skills/execute")
def run_agent_skill(req: ExecuteSkillRequest):
    """Executes a specific autonomous Agent Skill pipeline."""
    skill_id = req.skill_id
    params = req.params
    try:
        if skill_id == "podcast_host":
            return skills_engine.execute_podcast_skill(
                topic=params.get("topic", "Generative Voice AI"),
                turns_count=int(params.get("turns_count", 3)),
                model_id=params.get("model_id", "local_neural")
            )
        elif skill_id == "cross_lingual_translator":
            return skills_engine.execute_translation_skill(
                source_text=params.get("source_text", "Welcome to Chatterbox multilingual synthesis."),
                target_lang=params.get("target_language", "ja"),
                model_id=params.get("model_id", "local_neural")
            )
        elif skill_id == "document_narrator":
            return skills_engine.execute_narrator_skill(
                document_text=params.get("document_text", "Chapter 1: The Acoustic Horizon."),
                voice_style=params.get("voice_style", "Warm Conversational")
            )
        elif skill_id == "voice_director":
            return skills_engine.execute_voice_director_skill(
                script=params.get("script", "Listen closely, the future of voice has arrived."),
                mood=params.get("mood", "Dramatic Suspense")
            )
        elif skill_id == "creative_visual_synthesizer":
            prompt = params.get("prompt", params.get("user_input", "Neural Flow Matching Acoustic Landscape"))
            art = image_engine.generate_cinematic_artwork(prompt)
            narration_text = f"Visualizing scene: {prompt}. Rendered with high-dimensional acoustic flow matching."
            sr, wav_np, _ = synthesize(text=narration_text, language_id="en")
            filename = f"visual_{uuid.uuid4().hex[:6]}.wav"
            out_path = os.path.join(OUTPUT_DIR, filename)
            sf.write(out_path, wav_np, sr)
            art["audio_url"] = f"/static/audio_output/{filename}"
            art["narration"] = narration_text
            return art
        elif skill_id == "kids_rhymes":
            return skills_engine.execute_kids_rhymes_skill(
                theme=params.get("theme", "Rainbow Butterfly"),
                content_type=params.get("content_type", "Nursery Rhyme Song"),
                language=params.get("language", "en")
            )
        elif any(s["id"] == skill_id for s in skills_engine.list_skills()):
            return skills_engine.execute_custom_agent_skill(
                agent_id=skill_id,
                user_input=params.get("user_input", params.get("prompt", "Hello")),
                language=params.get("language", "en")
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unrecognized skill: '{skill_id}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill execution failed: {str(e)}")


class ImageGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Visual scene prompt")
    style: str = Field("cinematic", description="Style of visual artwork")


@app.post("/api/image/generate")
def generate_image_endpoint(req: ImageGenerateRequest):
    """Synthesizes high-impact cinematic artwork and visual scene compositions like Antigravity."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        return image_engine.generate_cinematic_artwork(req.prompt, req.style)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class InstallSkillRequest(BaseModel):
    skill_id: str


class CustomAgentRequest(BaseModel):
    name: str
    role: str = ""
    description: str = ""
    system_prompt: str = ""
    voice_archetype: str = "Warm Conversational"
    tools: List[str] = Field(default_factory=lambda: ["llm_reasoning", "voice_synthesis"])


class TeamSpeakRequest(BaseModel):
    member_id: str
    text: Optional[str] = None


@app.get("/api/team")
def get_team():
    """Returns team roster and AI personas."""
    return {"team": team_engine.get_team_roster()}


@app.post("/api/team/speak")
def team_member_speak(req: TeamSpeakRequest):
    """Auditions team persona or speaks custom quote."""
    member = team_engine.get_member_by_id(req.member_id)
    quote = req.text if req.text else member.get("sample_quote", "Hello, I am ready to assist you.")
    preset_id = member.get("preset_voice", "en_female")
    
    preset = next((p for p in PRESET_VOICES if p["id"] == preset_id), None)
    audio_prompt = preset["audio"] if preset else None
    
    import soundfile as sf
    sr, wav_np, _ = synthesize(text=quote, language_id="en", audio_prompt_path=audio_prompt)
    filename = f"team_{member['id']}_{uuid.uuid4().hex[:6]}.wav"
    out_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(out_path, wav_np, sr)
    return {
        "success": True,
        "member": member["name"],
        "text": quote,
        "audio_url": f"/static/audio_output/{filename}",
        "duration": round(len(wav_np) / sr, 2)
    }


@app.get("/api/skills/marketplace")
def get_skills_marketplace():
    """Returns installable skills from the Antigravity Agent Marketplace."""
    return {"marketplace": skills_engine.list_marketplace()}


@app.post("/api/skills/install")
def install_agent_skill(req: InstallSkillRequest):
    """Installs a skill into the active studio."""
    try:
        return skills_engine.install_skill(req.skill_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/skills/custom")
def define_custom_agent(req: CustomAgentRequest):
    """Creates a custom autonomous agent skill just like Antigravity."""
    try:
        return skills_engine.create_custom_agent(
            name=req.name,
            role=req.role,
            desc=req.description,
            system_prompt=req.system_prompt,
            voice_archetype=req.voice_archetype,
            tools=req.tools
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.get("/api/connectors/list")
def get_connectors():
    """Returns active state of all integration connectors."""
    return {"connectors": connectors.list_connectors()}


@app.post("/api/connectors/keys")
def set_connector_key(req: SetApiKeyRequest):
    """Saves provider API key."""
    llm_engine.set_api_key(req.provider_key, req.key_value)
    return {"success": True, "message": f"Updated {req.provider_key}"}


@app.post("/api/connectors/webhook")
def receive_inbound_webhook(payload: Dict[str, Any]):
    """Receives external webhook trigger and generates audio."""
    try:
        return connectors.handle_inbound_webhook(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/languages")
def get_languages():
    """Returns all 23 supported languages and their default sample prompts."""
    return {
        "languages": SUPPORTED_LANGUAGES,
        "samples": SAMPLE_CONFIG,
    }


@app.post("/api/chat")
def chat_synthesize(req: ChatSynthesisRequest):
    """
    Main conversational TTS synthesis endpoint for the Chat workspace.
    Synthesizes speech, saves output to static folder, and returns player metadata.
    """
    lang = req.language.lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language code '{lang}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        import soundfile as sf
        sr, wav_np, _ = synthesize(
            text=req.text.strip(),
            language_id=lang,
            audio_prompt_path=req.audio_prompt_url,
            exaggeration=req.exaggeration,
            temperature=req.temperature,
            cfg_weight=req.cfg_weight,
            seed=req.seed,
            t3_model=req.model_version,
        )

        audio_id = f"speech_{uuid.uuid4().hex[:10]}.wav"
        output_path = os.path.join(OUTPUT_DIR, audio_id)
        sf.write(output_path, wav_np, sr, format="WAV")

        duration = round(len(wav_np) / sr, 2)
        audio_url = f"/static/audio_output/{audio_id}"

        return {
            "success": True,
            "audio_url": audio_url,
            "filename": audio_id,
            "duration": duration,
            "sample_rate": sr,
            "text": req.text.strip(),
            "language": lang,
            "language_name": SUPPORTED_LANGUAGES.get(lang, lang),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/api/upload-voice")
async def upload_voice_clip(file: UploadFile = File(...)):
    """Upload a reference audio clip (WAV/MP3/FLAC/M4A) for zero-shot voice cloning."""
    allowed_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format. Allowed: {allowed_exts}")

    file_id = f"ref_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = os.path.join(UPLOADS_DIR, file_id)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "url": f"/static/uploads/{file_id}",
        "local_path": dest_path,
        "filename": file.filename,
    }


@app.post("/api/code/run")
def run_code_snippet(req: CodeRunRequest):
    """Executes a code test request and returns the resulting audio and terminal logs."""
    lang = req.language.lower()
    try:
        import soundfile as sf
        sr, wav_np, _ = synthesize(
            text=req.text,
            language_id=lang,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
        )

        audio_id = f"code_exec_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(OUTPUT_DIR, audio_id)
        sf.write(output_path, wav_np, sr, format="WAV")

        duration = round(len(wav_np) / sr, 2)
        logs = [
            f"[200 OK] Initialized Chatterbox Multilingual TTS (v3)",
            f"[INFO] Target Language: {lang.upper()} ({SUPPORTED_LANGUAGES.get(lang, lang)})",
            f"[INFO] Audio Rendered: {len(wav_np)} samples at {sr} Hz ({duration}s)",
            f"[SUCCESS] Audio written to {audio_id}"
        ]

        return {
            "success": True,
            "logs": logs,
            "audio_url": f"/static/audio_output/{audio_id}",
            "duration": duration,
        }
    except Exception as e:
        return {
            "success": False,
            "logs": [f"[ERROR] Execution failed: {str(e)}"],
            "error": str(e)
        }


# Autonomous Worker endpoints
@app.post("/api/workers/batch")
def submit_batch_job(req: BatchWorkerRequest):
    """Enqueues a batch generation job to background workers."""
    job_id = workers.submit_batch_tts_worker(
        texts=req.texts,
        language=req.language,
        ref_audio=req.ref_audio,
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
    )
    return {"success": True, "job_id": job_id, "message": "Batch worker job enqueued."}


@app.post("/api/workers/dubbing")
def submit_dubbing_job(req: DubbingWorkerRequest):
    """Enqueues a cross-lingual dubbing job."""
    job_id = workers.submit_dubbing_worker(
        source_text=req.source_text,
        target_languages=req.target_languages,
        ref_audio=req.ref_audio,
    )
    return {"success": True, "job_id": job_id, "message": "Dubbing worker job enqueued."}


@app.post("/api/workers/profile")
def profile_voice(file: UploadFile = File(...)):
    """Profiles and analyzes an uploaded voice clip."""
    ext = os.path.splitext(file.filename)[1].lower()
    file_id = f"profile_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = os.path.join(UPLOADS_DIR, file_id)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = workers.submit_voice_profiler_worker(dest_path, file.filename)
    return {"success": True, "job_id": job_id, "file_url": f"/static/uploads/{file_id}"}


PRESET_VOICES = [
    {"id": "en_female", "name": "Sarah (English)", "lang": "en", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac", "avatar": "👩🏼", "desc": "Clear, friendly conversational tone"},
    {"id": "fr_female", "name": "Camille (French)", "lang": "fr", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac", "avatar": "👩🏻", "desc": "Warm, authentic Parisian accent"},
    {"id": "es_female", "name": "Elena (Spanish)", "lang": "es", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac", "avatar": "👩🏽", "desc": "Energetic, expressive Castilian & LatAm"},
    {"id": "de_female", "name": "Greta (German)", "lang": "de", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac", "avatar": "👱🏻‍♀️", "desc": "Crisp, natural narration"},
    {"id": "it_male", "name": "Marco (Italian)", "lang": "it", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac", "avatar": "👨🏻", "desc": "Rich, melodic Italian timbre"},
    {"id": "pt_male", "name": "Thiago (Portuguese)", "lang": "pt", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac", "avatar": "👨🏽", "desc": "Smooth Brazilian inflection"},
    {"id": "zh_female", "name": "Mei (Chinese)", "lang": "zh", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac", "avatar": "👩🏻", "desc": "Fluid Mandarin pronunciation"},
    {"id": "ja_female", "name": "Yuki (Japanese)", "lang": "ja", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac", "avatar": "👧🏻", "desc": "Gentle, natural Tokyo dialect"},
    {"id": "hi_female", "name": "Aanya (Hindi)", "lang": "hi", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac", "avatar": "👩🏾", "desc": "Graceful Hindi rhythm and clarity"},
    {"id": "ru_male", "name": "Dmitri (Russian)", "lang": "ru", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac", "avatar": "👨🏼", "desc": "Authoritative, deep resonance"},
    {"id": "ar_female", "name": "Layla (Arabic)", "lang": "ar", "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac", "avatar": "🧕🏽", "desc": "Poised Modern Standard Arabic"},
]


@app.get("/api/presets")
def get_preset_voices():
    """Returns curated preset voices for 1-click voice cloning."""
    return {"presets": PRESET_VOICES}


@app.get("/api/workers/jobs")
def list_worker_jobs():
    """Returns all queued and completed background worker jobs."""
    return {"jobs": workers.get_all_jobs()}


@app.get("/api/workers/jobs/{job_id}")
def get_worker_job_status(job_id: str):
    """Returns the status and logs for a specific worker job."""
    job = workers.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}


@app.delete("/api/workers/jobs/{job_id}")
def delete_worker_job(job_id: str):
    """Deletes a job from the queue."""
    deleted = workers.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"success": True, "message": f"Job {job_id} deleted."}


@app.post("/api/workers/jobs/clear")
def clear_completed_worker_jobs():
    """Clears all completed or failed jobs."""
    workers.clear_completed_jobs()
    return {"success": True, "message": "Cleared completed jobs."}


def find_free_port(default_port: int = 8080) -> int:
    import socket
    port = int(os.getenv("PORT", default_port))
    for p in range(port, port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return port


if __name__ == "__main__":
    import uvicorn
    port = find_free_port(8080)
    print(f"[*] Starting Chatterbox Z.ai Server on http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
