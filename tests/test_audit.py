"""
test_audit.py
Comprehensive Verification & Audit Test Suite for Chatterbox Multilingual TTS Platform.
Audits:
1. Multilingual Support & Language Matrix (23 languages)
2. macOS Acceleration & Fallback Engine (MPS/CPU)
3. Payment Gateway, Pricing Plans, Scan-to-Pay QR Orders, and Credit Settlement
4. Background Autonomous Worker Queue & Job Lifecycle
5. Full FastAPI REST API Endpoints via TestClient
6. Synthesis Audio Output & Serialization
"""
import os
import sys
import unittest
import uuid
import json
from unittest.mock import patch
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mac_patch
import billing
import workers
from tts_engine import SUPPORTED_LANGUAGES, SAMPLE_CONFIG, get_target_device
from server import app
from fastapi.testclient import TestClient


class TestAuditChatterboxPlatform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # ==========================================
    # 1. MULTILINGUAL & ENGINE AUDIT
    # ==========================================
    def test_01_supported_languages_matrix(self):
        """Verify all 23 official languages are registered with valid codes and names."""
        expected_languages = [
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
            "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
            "sw", "tr", "zh"
        ]
        self.assertEqual(len(SUPPORTED_LANGUAGES), 23, "Chatterbox must support exactly 23 languages.")
        for lang_code in expected_languages:
            self.assertIn(lang_code, SUPPORTED_LANGUAGES, f"Language '{lang_code}' must be in SUPPORTED_LANGUAGES")
            self.assertTrue(len(SUPPORTED_LANGUAGES[lang_code]) > 0, f"Language '{lang_code}' has empty label")

    def test_02_sample_prompts_configured(self):
        """Verify sample prompts are configured for major languages."""
        for lang in ["en", "es", "fr", "de", "it", "pt", "hi", "zh", "ja"]:
            self.assertIn(lang, SAMPLE_CONFIG, f"Sample prompt missing for {lang}")
            self.assertIn("text", SAMPLE_CONFIG[lang], f"Sample text missing for {lang}")
            self.assertGreater(len(SAMPLE_CONFIG[lang]["text"]), 5, f"Sample prompt too short for {lang}")
            self.assertIn("audio", SAMPLE_CONFIG[lang], f"Sample audio missing for {lang}")

    def test_03_device_detection_and_mps_fallback(self):
        """Audit device selection on macOS (MPS or CPU) and environment flags."""
        device = get_target_device()
        self.assertIn(device, ["mps", "cpu", "cuda"], f"Device '{device}' is not a recognized device.")
        
        # Verify macOS MPS fallback env flag is active
        fallback_flag = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        self.assertEqual(fallback_flag, "1", "PYTORCH_ENABLE_MPS_FALLBACK must be set to '1' for Apple Silicon/macOS safety.")

    # ==========================================
    # 2. PAYMENT GATEWAY & SCAN-TO-PAY AUDIT
    # ==========================================
    def test_04_billing_plans_specification(self):
        """Audit billing plans for pricing, credit tiers, and feature sets."""
        self.assertGreaterEqual(len(billing.PLANS), 3, "Must offer at least 3 distinct pricing tiers.")
        plan_ids = [p["id"] for p in billing.PLANS]
        self.assertIn("plan_starter", plan_ids)
        self.assertIn("plan_pro", plan_ids)
        self.assertIn("plan_enterprise", plan_ids)

        for plan in billing.PLANS:
            self.assertGreater(plan["credits"], 0, "Credits must be positive.")
            self.assertGreater(plan["price_usd"], 0, "USD Price must be positive.")
            self.assertGreater(plan["price_inr"], 0, "INR Price must be positive.")
            self.assertTrue(len(plan["features"]) > 0, "Plan must define features.")

    def test_05_scan_to_pay_order_lifecycle(self):
        """Audit Scan to Pay order creation, UPI payload generation, and balance settlement."""
        initial_balance = billing.get_balance()["credits"]

        # 1. Create order for starter pack
        order = billing.create_scan_to_pay_order("plan_starter")
        order_id = order["order_id"]

        self.assertTrue(order_id.startswith("ORDER-"), "Order ID must follow standard prefix.")
        self.assertEqual(order["status"], "pending")
        self.assertEqual(order["credits"], 10000)
        self.assertEqual(order["amount_inr"], 750)
        self.assertIn("upi://pay", order["upi_url"], "Order must contain a valid UPI deep-link URI.")
        self.assertIn(order_id, order["upi_url"], "UPI URI must embed the unique transaction reference ID.")

        # 2. Verify payment and settlement
        verify_res = billing.verify_order_payment(order_id)
        self.assertTrue(verify_res["success"])
        self.assertEqual(verify_res["order"]["status"], "completed")

        new_balance = billing.get_balance()["credits"]
        self.assertEqual(new_balance, initial_balance + 10000, "User balance must increase by order credit quantity.")

        # 3. Double-settlement idempotency check (should not grant duplicate credits)
        verify_again = billing.verify_order_payment(order_id)
        self.assertTrue(verify_again["success"])
        self.assertEqual(billing.get_balance()["credits"], new_balance, "Duplicate payment verification must not add extra credits.")

    def test_06_billing_deduct_and_invalid_order(self):
        """Audit credit deduction and error handling for invalid plan IDs."""
        current = billing.get_balance()["credits"]
        self.assertTrue(billing.deduct_credits(100), "Should successfully deduct available credits.")
        self.assertEqual(billing.get_balance()["credits"], current - 100)

        # Attempt to deduct excessive amount
        self.assertFalse(billing.deduct_credits(999999999), "Should refuse deduction exceeding balance.")

        # Invalid plan creation
        with self.assertRaises(ValueError):
            billing.create_scan_to_pay_order("non_existent_plan_xyz")

        # Unknown order verification
        with self.assertRaises(ValueError):
            billing.verify_order_payment("ORDER-UNKNOWN-0000")

    # ==========================================
    # 3. AUTONOMOUS WORKERS AUDIT
    # ==========================================
    @patch("workers.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_07_worker_job_queue_and_cleanup(self, mock_synth):
        """Audit autonomous background worker queue submission, status tracking, and deletion."""
        # Submit batch job
        job_id = workers.submit_batch_tts_worker(
            texts=["Sentence 1 for testing.", "Sentence 2 for testing."],
            language="en",
            exaggeration=0.5,
            cfg_weight=0.5
        )
        self.assertTrue(job_id.startswith("worker-"))

        job = workers.get_job(job_id)
        self.assertIsNotNone(job, "Job must exist in registry.")
        self.assertEqual(job["type"], "batch_tts")
        self.assertIn(job["status"], ["queued", "running", "completed"])

        # Check job logs
        workers.add_job_log(job_id, "Test audit log entry.")
        updated_job = workers.get_job(job_id)
        self.assertTrue(any("Test audit log entry." in l for l in updated_job["logs"]))

        # Verify listing
        all_jobs = workers.get_all_jobs()
        self.assertTrue(any(j["id"] == job_id for j in all_jobs))

        # Delete job
        deleted = workers.delete_job(job_id)
        self.assertTrue(deleted, "Job should be successfully deleted.")
        self.assertIsNone(workers.get_job(job_id))

    @patch("workers.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_08_dubbing_worker_submission(self, mock_synth):
        """Audit cross-lingual dubbing worker job registration."""
        job_id = workers.submit_dubbing_worker(
            source_text="Welcome to Chatterbox multilingual speech synthesis.",
            target_languages=["fr", "es", "de"],
            ref_audio=None
        )
        self.assertTrue(job_id.startswith("worker-"))
        job = workers.get_job(job_id)
        self.assertEqual(job["type"], "dubbing")
        self.assertEqual(job["meta"]["languages"], ["fr", "es", "de"])

        # Cleanup
        workers.delete_job(job_id)

    # ==========================================
    # 4. FASTAPI REST API ENDPOINTS AUDIT
    # ==========================================
    def test_09_api_status(self):
        """GET /api/status returns engine metadata and readiness."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["engine"], "Chatterbox Multilingual TTS")
        self.assertEqual(data["supported_languages_count"], 23)
        self.assertEqual(data["status"], "ready")

    def test_10_api_languages(self):
        """GET /api/languages returns 23 languages and sample prompts."""
        res = self.client.get("/api/languages")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["languages"]), 23)
        self.assertIn("en", data["languages"])
        self.assertIn("ja", data["languages"])

    def test_11_api_presets(self):
        """GET /api/presets returns curated reference voices."""
        res = self.client.get("/api/presets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("presets", data)
        self.assertGreaterEqual(len(data["presets"]), 10)
        for preset in data["presets"]:
            self.assertIn("id", preset)
            self.assertIn("audio", preset)
            self.assertIn("name", preset)

    def test_12_api_billing_endpoints(self):
        """Audit REST endpoints: /api/billing/plans, /api/billing/balance, create-order, and verify-payment."""
        # 1. Plans
        p_res = self.client.get("/api/billing/plans")
        self.assertEqual(p_res.status_code, 200)
        self.assertIn("plans", p_res.json())

        # 2. Balance
        b_res = self.client.get("/api/billing/balance")
        self.assertEqual(b_res.status_code, 200)
        current_credits = b_res.json()["credits"]

        # 3. Create order
        order_res = self.client.post("/api/billing/create-order", json={"plan_id": "plan_pro"})
        self.assertEqual(order_res.status_code, 200)
        order_data = order_res.json()["order"]
        order_id = order_data["order_id"]
        self.assertEqual(order_data["credits"], 50000)

        # 4. Verify order
        pay_res = self.client.post("/api/billing/verify-payment", json={"order_id": order_id})
        self.assertEqual(pay_res.status_code, 200)
        pay_data = pay_res.json()
        self.assertTrue(pay_data["success"])
        self.assertEqual(pay_data["new_balance"], current_credits + 50000)

        # 5. Invalid plan error handling
        err_res = self.client.post("/api/billing/create-order", json={"plan_id": "bad_plan"})
        self.assertEqual(err_res.status_code, 400)

        # 6. Unknown order verification error handling
        bad_verify = self.client.post("/api/billing/verify-payment", json={"order_id": "ORDER-DOES-NOT-EXIST"})
        self.assertEqual(bad_verify.status_code, 404)

    @patch("workers.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_13_api_workers_endpoints(self, mock_synth):
        """Audit REST endpoints: /api/workers/jobs, batch submit, dubbing submit, delete, and clear."""
        # 1. Submit batch via API
        batch_res = self.client.post("/api/workers/batch", json={
            "texts": ["Hello world", "Audio synthesis in progress"],
            "language": "en",
            "exaggeration": 0.5,
            "cfg_weight": 0.5
        })
        self.assertEqual(batch_res.status_code, 200)
        batch_job_id = batch_res.json()["job_id"]

        # 2. Submit dubbing via API
        dub_res = self.client.post("/api/workers/dubbing", json={
            "source_text": "Multilingual translation test",
            "target_languages": ["fr", "es"]
        })
        self.assertEqual(dub_res.status_code, 200)
        dub_job_id = dub_res.json()["job_id"]

        # 3. List jobs
        list_res = self.client.get("/api/workers/jobs")
        self.assertEqual(list_res.status_code, 200)
        jobs = list_res.json()["jobs"]
        job_ids = [j["id"] for j in jobs]
        self.assertIn(batch_job_id, job_ids)
        self.assertIn(dub_job_id, job_ids)

        # 4. Get specific job
        job_res = self.client.get(f"/api/workers/jobs/{batch_job_id}")
        self.assertEqual(job_res.status_code, 200)
        self.assertEqual(job_res.json()["job"]["id"], batch_job_id)

        # 5. Delete specific job
        del_res = self.client.delete(f"/api/workers/jobs/{batch_job_id}")
        self.assertEqual(del_res.status_code, 200)

        # 6. Clear completed
        clear_res = self.client.post("/api/workers/jobs/clear")
        self.assertEqual(clear_res.status_code, 200)

    def test_14_api_chat_validation(self):
        """Audit /api/chat input validation for unsupported languages and empty texts."""
        # Empty text
        res1 = self.client.post("/api/chat", json={"text": "", "language": "en"})
        self.assertEqual(res1.status_code, 400)

        # Unsupported language
        res2 = self.client.post("/api/chat", json={"text": "Hello", "language": "unsupported_lang"})
        self.assertEqual(res2.status_code, 400)

    def test_15_index_html_served(self):
        """Audit that GET / serves the chat.z.ai interface."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        content = res.text
        self.assertIn("Chatterbox", content)
        self.assertIn("Scan to Pay", content)
        self.assertIn("Autonomous Workers", content)

    @patch("server.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_16_code_run_endpoint(self, mock_synth):
        """Audit /api/code/run endpoint for Code workspace."""
        res = self.client.post("/api/code/run", json={
            "language": "ja",
            "text": "こんにちは世界",
            "exaggeration": 0.6,
            "cfg_weight": 0.4
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("audio_url", data)
        self.assertIn("logs", data)
        self.assertTrue(any("200 OK" in l for l in data["logs"]))

    @patch("server.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_17_chat_synthesize_endpoint(self, mock_synth):
        """Audit /api/chat endpoint for Chat workspace."""
        res = self.client.post("/api/chat", json={
            "text": "Hello, this is Chatterbox AI testing synthesis.",
            "language": "en",
            "model_version": "v3"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("audio_url", data)
        self.assertEqual(data["language"], "en")

    def test_18_audio_file_integrity(self):
        """Audit synthesized WAV files for header, format, and sample rate integrity."""
        import soundfile as sf
        output_dir = os.path.join(PROJECT_ROOT, "static", "audio_output")
        wav_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav")]
        self.assertGreater(len(wav_files), 0, "There should be synthesized audio files generated by tests.")
        for wav_file in wav_files[:5]:  # verify sample of files
            data, sr = sf.read(wav_file)
            self.assertEqual(sr, 24000, f"Sample rate must be 24kHz, got {sr}")
            self.assertGreater(len(data), 0, f"WAV file {wav_file} contains no audio samples.")

    def test_19_copilot_chat_endpoint(self):
        """Audit /api/copilot/chat endpoint grounded on project architecture."""
        # 1. Ask about 23 languages
        res = self.client.post("/api/copilot/chat", json={"query": "What 23 languages are supported?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["category"], "languages")
        self.assertIn("23 Total", data["answer"])

        # 2. Ask about voice cloning
        res_clone = self.client.post("/api/copilot/chat", json={"query": "How do I clone a voice?"})
        self.assertEqual(res_clone.status_code, 200)
        self.assertEqual(res_clone.json()["category"], "cloning")

        # 3. Empty query rejection
        err_res = self.client.post("/api/copilot/chat", json={"query": "   "})
        self.assertEqual(err_res.status_code, 400)

    def test_20_official_upi_id_configured(self):
        """Audit that official merchant UPI ID 9994152888-4#ybl is active in billing engine."""
        order = billing.create_scan_to_pay_order("plan_starter")
        self.assertEqual(order["pay_address"], "9994152888-4#ybl")
        self.assertIn("pa=9994152888-4%23ybl", order["upi_url"])

    def test_21_llm_models_and_generation(self):
        """Audit /api/llm/models and /api/llm/generate endpoints."""
        # 1. Models matrix
        res = self.client.get("/api/llm/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertGreaterEqual(len(data["models"]), 6)
        model_ids = [m["id"] for m in data["models"]]
        self.assertIn("local_neural", model_ids)
        self.assertIn("gemini_pro", model_ids)
        self.assertIn("deepseek_v3", model_ids)
        self.assertNotIn("claude_35", model_ids)

        # 2. Text generation on Local Neural AI
        gen_res = self.client.post("/api/llm/generate", json={
            "prompt": "What is the capital of France?",
            "model_id": "local_neural"
        })
        self.assertEqual(gen_res.status_code, 200)
        gen_data = gen_res.json()
        self.assertTrue(gen_data["success"])
        self.assertIn("text", gen_data)

        # 3. Validation on empty prompt
        err_res = self.client.post("/api/llm/generate", json={
            "prompt": "   ",
            "model_id": "local_neural"
        })
        self.assertEqual(err_res.status_code, 400)

    @patch("tts_engine.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    @patch("skills_engine.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_22_autonomous_skills_execution(self, mock_skill_synth, mock_tts_synth):
        """Audit /api/skills/list and /api/skills/execute endpoints."""
        # 1. List skills
        res = self.client.get("/api/skills/list")
        self.assertEqual(res.status_code, 200)
        skills = res.json()["skills"]
        skill_ids = [s["id"] for s in skills]
        self.assertIn("podcast_host", skill_ids)
        self.assertIn("cross_lingual_translator", skill_ids)
        self.assertIn("document_narrator", skill_ids)
        self.assertIn("voice_director", skill_ids)

        # 2. Execute podcast skill
        pod_res = self.client.post("/api/skills/execute", json={
            "skill_id": "podcast_host",
            "params": {"topic": "Generative Speech", "turns_count": 2, "model_id": "local_neural"}
        })
        self.assertEqual(pod_res.status_code, 200)
        pod_data = pod_res.json()
        self.assertTrue(pod_data["success"])
        self.assertIn("script", pod_data)
        self.assertIn("audio_url", pod_data)

        # 3. Execute translator skill
        trans_res = self.client.post("/api/skills/execute", json={
            "skill_id": "cross_lingual_translator",
            "params": {"source_text": "Hello world", "target_language": "ja"}
        })
        self.assertEqual(trans_res.status_code, 200)
        self.assertTrue(trans_res.json()["success"])

        # 4. Unknown skill rejection
        err_skill = self.client.post("/api/skills/execute", json={
            "skill_id": "invalid_skill_xyz",
            "params": {}
        })
        self.assertEqual(err_skill.status_code, 400)

    @patch("tts_engine.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_23_connectors_catalog_and_webhook(self, mock_synth):
        """Audit /api/connectors/list, key registration, and inbound webhook synthesis."""
        # 1. List connectors
        res = self.client.get("/api/connectors/list")
        self.assertEqual(res.status_code, 200)
        connectors_list = res.json()["connectors"]
        conn_ids = [c["id"] for c in connectors_list]
        self.assertIn("gemini_connector", conn_ids)
        self.assertIn("openai_connector", conn_ids)
        self.assertIn("deepseek_connector", conn_ids)
        self.assertNotIn("anthropic_connector", conn_ids)
        self.assertIn("webhook_connector", conn_ids)

        # 2. Save API key
        key_res = self.client.post("/api/connectors/keys", json={
            "provider_key": "GEMINI_API_KEY",
            "key_value": "test-key-audit-12345"
        })
        self.assertEqual(key_res.status_code, 200)
        self.assertTrue(key_res.json()["success"])

        # 3. Inbound webhook triggering synthesis
        webhook_res = self.client.post("/api/connectors/webhook", json={
            "event": "voice.synthesize",
            "text": "Audit test inbound webhook payload.",
            "language": "en"
        })
        self.assertEqual(webhook_res.status_code, 200)
        w_data = webhook_res.json()
        self.assertTrue(w_data["success"])
        self.assertIn("audio_url", w_data)

    @patch("server.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_24_team_roster_and_voice_audition(self, mock_synth):
        """Audit /api/team roster, human founder persona, and voice audition synthesis."""
        # 1. Team roster endpoint
        res = self.client.get("/api/team")
        self.assertEqual(res.status_code, 200)
        team = res.json()["team"]
        self.assertGreaterEqual(len(team), 4)

        founder = next((m for m in team if m["id"] == "char_founder"), None)
        self.assertIsNotNone(founder)
        self.assertEqual(founder["name"], "G.S.")
        self.assertIn("Flow Matching", founder["specialties"])
        self.assertTrue(os.path.exists(founder["avatar"].lstrip("/")))

        # 2. Character audition speak endpoint
        speak_res = self.client.post("/api/team/speak", json={
            "member_id": "char_founder",
            "text": "Voice audition test line."
        })
        self.assertEqual(speak_res.status_code, 200)
        data = speak_res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["member"], "G.S.")
        self.assertIn("audio_url", data)

    @patch("skills_engine.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_25_antigravity_skills_marketplace_and_custom_agent(self, mock_synth):
        """Audit Antigravity Skills Marketplace listing, installation, and custom subagent definition."""
        # 1. Marketplace catalog
        res = self.client.get("/api/skills/marketplace")
        self.assertEqual(res.status_code, 200)
        marketplace = res.json()["marketplace"]
        self.assertGreaterEqual(len(marketplace), 3)

        # 2. Install skill
        install_res = self.client.post("/api/skills/install", json={"skill_id": "interactive_tutor"})
        self.assertEqual(install_res.status_code, 200)
        self.assertTrue(install_res.json()["success"])

        # 3. Define custom autonomous subagent
        custom_res = self.client.post("/api/skills/custom", json={
            "name": "Audit Test Storyteller",
            "role": "Narrative Synthesis",
            "description": "Tests custom Antigravity subagent definition",
            "system_prompt": "You are a test agent.",
            "voice_archetype": "Epic Cinematic",
            "tools": ["llm_reasoning", "voice_synthesis"]
        })
        self.assertEqual(custom_res.status_code, 200)
        custom_data = custom_res.json()
        self.assertTrue(custom_data["success"])
        new_skill_id = custom_data["skill"]["id"]

        # 4. Execute custom subagent
        exec_res = self.client.post("/api/skills/execute", json={
            "skill_id": new_skill_id,
            "params": {"prompt": "Tell a brief audit tale"}
        })
        self.assertEqual(exec_res.status_code, 200)
        self.assertTrue(exec_res.json()["success"])

    def test_26_image_generation_and_visual_artwork(self):
        """Audit Antigravity-style creative vision & image generation endpoint."""
        res = self.client.post("/api/image/generate", json={
            "prompt": "Cyberpunk Neural Soundwave Horizon",
            "style": "cinematic"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("image_url", data)
        self.assertIn("filename", data)
        self.assertTrue(data["image_url"].endswith(".svg"))
        # Verify file exists on disk
        img_path = os.path.join("static", "img_output", data["filename"])
        self.assertTrue(os.path.exists(img_path))

    @patch("skills_engine.synthesize", return_value=(24000, np.zeros(24000, dtype=np.float32), None))
    def test_27_kids_entertainment_skill(self, mock_synth):
        """Audit Khyathi.Sri Kids Songs & Story Studio with Key Secure Foundation partner."""
        res = self.client.post("/api/skills/execute", json={
            "skill_id": "kids_rhymes",
            "params": {
                "theme": "Smiling Star in Midnight Sky",
                "content_type": "Nursery Rhyme Song",
                "language": "en"
            }
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["character"], "Khyathi Sri")
        self.assertIn("image_url", data)
        self.assertIn("audio_url", data)
        self.assertEqual(data["partner"], "Key Secure Foundation")

    def test_28_auth_system(self):
        """Audit Authentication system: register, login, session validation, logout."""
        test_email = f"audit_{uuid.uuid4().hex[:6]}@khyathi.sri"

        # 1. Registration
        reg_res = self.client.post("/api/auth/register", json={
            "email": test_email,
            "password": "auditpassword123",
            "name": "Audit Tester",
            "role": "PG Researcher"
        })
        self.assertEqual(reg_res.status_code, 200)
        reg_data = reg_res.json()
        self.assertTrue(reg_data["success"])
        self.assertIn("token", reg_data)
        token = reg_data["token"]
        self.assertEqual(reg_data["user"]["role"], "PG Researcher")

        # 2. Get current user via Token header
        me_res = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.json()
        self.assertTrue(me_data["success"])
        self.assertEqual(me_data["user"]["email"], test_email)

        # 3. Login
        login_res = self.client.post("/api/auth/login", json={
            "email": test_email,
            "password": "auditpassword123"
        })
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.json()
        self.assertTrue(login_data["success"])

        # 4. Invalid Login
        fail_res = self.client.post("/api/auth/login", json={
            "email": test_email,
            "password": "wrongpassword"
        })
        self.assertEqual(fail_res.status_code, 401)

        # 5. Logout
        logout_res = self.client.post("/api/auth/logout", json={"token": token})
        self.assertEqual(logout_res.status_code, 200)
        self.assertTrue(logout_res.json()["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

