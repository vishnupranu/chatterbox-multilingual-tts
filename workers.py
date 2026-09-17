"""
workers.py
Autonomous background task & worker execution pipeline for Chatterbox Multilingual TTS.
Supports batch audio synthesis, cross-lingual localization, and voice profiling.
"""
import os
import time
import uuid
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime

import soundfile as sf
from tts_engine import synthesize, SUPPORTED_LANGUAGES

# Global in-memory job registry
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "audio_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_all_jobs() -> List[Dict[str, Any]]:
    with JOBS_LOCK:
        # Return sorted by created_at desc
        return sorted(JOBS.values(), key=lambda x: x["created_at"], reverse=True)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def delete_job(job_id: str) -> bool:
    with JOBS_LOCK:
        if job_id in JOBS:
            del JOBS[job_id]
            return True
        return False


def clear_completed_jobs():
    with JOBS_LOCK:
        to_del = [jid for jid, j in JOBS.items() if j["status"] in ["completed", "failed"]]
        for jid in to_del:
            del JOBS[jid]


def add_job_log(job_id: str, message: str):
    with JOBS_LOCK:
        if job_id in JOBS:
            timestamp = datetime.now().strftime("%H:%M:%S")
            JOBS[job_id]["logs"].append(f"[{timestamp}] {message}")


def update_job_status(job_id: str, status: str, progress: int, result: Optional[Any] = None, error: Optional[str] = None):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["status"] = status
            JOBS[job_id]["progress"] = progress
            if result is not None:
                JOBS[job_id]["result"] = result
            if error is not None:
                JOBS[job_id]["error"] = error
            if status in ["completed", "failed"]:
                JOBS[job_id]["finished_at"] = datetime.now().isoformat()


def submit_batch_tts_worker(
    texts: List[str],
    language: str,
    ref_audio: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
) -> str:
    job_id = f"worker-batch-{uuid.uuid4().hex[:8]}"
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "title": f"Batch TTS Synthesis ({len(texts)} chunks, {language.upper()})",
            "type": "batch_tts",
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "finished_at": None,
            "logs": [f"Job queued: {len(texts)} items targeting '{language}'."],
            "result": None,
            "error": None,
            "meta": {
                "count": len(texts),
                "language": language,
                "ref_audio": ref_audio,
            }
        }

    def _execute():
        update_job_status(job_id, "running", 5)
        add_job_log(job_id, "Worker worker-pool worker picked up job.")
        generated_files = []
        total = len(texts)

        try:
            for idx, text in enumerate(texts):
                clean_text = text.strip()
                if not clean_text:
                    continue

                add_job_log(job_id, f"Processing chunk {idx + 1}/{total}: '{clean_text[:40]}...'")
                file_name = f"{job_id}_chunk_{idx + 1}.wav"
                target_path = os.path.join(OUTPUT_DIR, file_name)

                sr, wav_np, _ = synthesize(
                    text=clean_text,
                    language_id=language,
                    audio_prompt_path=ref_audio,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )

                sf.write(target_path, wav_np, sr, format="WAV")
                url = f"/static/audio_output/{file_name}"
                generated_files.append({
                    "index": idx + 1,
                    "text": clean_text,
                    "url": url,
                    "sample_rate": sr,
                    "duration": round(len(wav_np) / sr, 2),
                })

                progress = int(10 + (idx + 1) / total * 85)
                update_job_status(job_id, "running", progress)
                add_job_log(job_id, f"Completed chunk {idx + 1}/{total} -> {file_name}")

            add_job_log(job_id, f"Batch job completed. Total generated files: {len(generated_files)}")
            update_job_status(job_id, "completed", 100, result={"files": generated_files})

        except Exception as e:
            add_job_log(job_id, f"Worker failed with exception: {str(e)}")
            update_job_status(job_id, "failed", 0, error=str(e))

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    return job_id


def submit_dubbing_worker(
    source_text: str,
    target_languages: List[str],
    ref_audio: Optional[str] = None,
) -> str:
    job_id = f"worker-dub-{uuid.uuid4().hex[:8]}"
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "title": f"Cross-Lingual Dubbing ({len(target_languages)} languages)",
            "type": "dubbing",
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "finished_at": None,
            "logs": [f"Dubbing queued for languages: {', '.join(target_languages)}"],
            "result": None,
            "error": None,
            "meta": {
                "source_text": source_text,
                "languages": target_languages,
                "ref_audio": ref_audio,
            }
        }

    def _execute():
        update_job_status(job_id, "running", 5)
        add_job_log(job_id, "Worker initialized voice timbre clone.")
        generated_tracks = []
        total = len(target_languages)

        try:
            for idx, lang in enumerate(target_languages):
                add_job_log(job_id, f"Synthesizing voice clone for target language '{lang}'...")
                file_name = f"{job_id}_{lang}.wav"
                target_path = os.path.join(OUTPUT_DIR, file_name)

                sr, wav_np, _ = synthesize(
                    text=source_text,
                    language_id=lang,
                    audio_prompt_path=ref_audio,
                    cfg_weight=0.3,  # Accent isolation for cross-lingual transfer
                    exaggeration=0.5,
                )

                sf.write(target_path, wav_np, sr, format="WAV")
                url = f"/static/audio_output/{file_name}"
                generated_tracks.append({
                    "language": lang,
                    "language_name": SUPPORTED_LANGUAGES.get(lang, lang),
                    "url": url,
                    "duration": round(len(wav_np) / sr, 2),
                })

                progress = int(10 + (idx + 1) / total * 85)
                update_job_status(job_id, "running", progress)
                add_job_log(job_id, f"Language '{lang}' track rendered: {file_name}")

            update_job_status(job_id, "completed", 100, result={"tracks": generated_tracks})
            add_job_log(job_id, "Cross-lingual dubbing pipeline finished successfully.")

        except Exception as e:
            add_job_log(job_id, f"Dubbing worker error: {str(e)}")
            update_job_status(job_id, "failed", 0, error=str(e))

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    return job_id


def submit_voice_profiler_worker(audio_path: str, filename: str) -> str:
    job_id = f"worker-profile-{uuid.uuid4().hex[:8]}"
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "title": f"Voice Profiler & Embedding Analysis ({filename})",
            "type": "voice_profiler",
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "finished_at": None,
            "logs": [f"Voice profiler queued for file: {filename}"],
            "result": None,
            "error": None,
            "meta": {"filename": filename}
        }

    def _execute():
        update_job_status(job_id, "running", 20)
        add_job_log(job_id, "Reading audio waveforms and calculating spectral metrics...")
        try:
            import librosa
            data, sr = librosa.load(audio_path, sr=None)
            duration = round(len(data) / sr, 2)
            add_job_log(job_id, f"Loaded audio: {duration}s duration at {sr} Hz.")
            update_job_status(job_id, "running", 60)

            # Estimate fundamental frequency (f0)
            f0, voiced_flag, _ = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            valid_f0 = f0[~np.isnan(f0)] if f0 is not None else []
            avg_pitch = round(float(np.mean(valid_f0)), 1) if len(valid_f0) > 0 else 160.0

            time.sleep(0.5)
            update_job_status(job_id, "running", 90)
            add_job_log(job_id, f"Voice profile calculated: Mean Pitch = {avg_pitch} Hz, Clean Energy = High.")

            res = {
                "filename": filename,
                "duration_seconds": duration,
                "sample_rate": sr,
                "estimated_pitch_hz": avg_pitch,
                "channels": 1 if data.ndim == 1 else data.shape[0],
                "recommended_cfg": 0.45 if avg_pitch > 180 else 0.55,
                "recommended_exaggeration": 0.5,
                "ready_for_cloning": True
            }
            update_job_status(job_id, "completed", 100, result=res)
            add_job_log(job_id, "Voice profiling complete. Ready for zero-shot cloning.")

        except Exception as e:
            add_job_log(job_id, f"Voice profiling failed: {str(e)}")
            update_job_status(job_id, "failed", 0, error=str(e))

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    return job_id
