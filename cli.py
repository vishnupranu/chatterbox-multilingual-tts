#!/usr/bin/env python3
"""
cli.py
Command-Line Interface & Interactive Terminal with Integrated AITalk Copilot.
Supports direct synthesis, interactive REPL, and project-trained AI assistant queries.
"""
import argparse
import sys
import os
from tts_engine import (
    SUPPORTED_LANGUAGES,
    synthesize,
    save_audio,
    get_target_device,
)
import copilot
import billing

# Enforce macOS MPS safety
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def list_languages():
    print("\n🌐 Supported Languages (23 Total):")
    print("=" * 45)
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        print(f"  {code:<6} -> {name}")
    print("=" * 45)


def run_interactive_terminal(ai_mode: bool = False):
    """Interactive terminal shell with integrated AITalk Copilot."""
    print("==========================================================")
    print("   CHATTERBOX MULTILINGUAL TTS - INTERACTIVE TERMINAL     ")
    print("   With Built-In Project-Trained AITalk Copilot           ")
    print("==========================================================")
    print("Available Commands:")
    print("  /synth <text>        -> Synthesize speech in active language")
    print("  /lang <code>         -> Switch language (e.g. /lang ja, /lang fr)")
    print("  /clone <audio_path>  -> Set reference audio for zero-shot cloning")
    print("  /languages           -> List all 23 supported languages")
    print("  /plans               -> View Scan-to-Pay pricing & UPI ID")
    print("  /ask <question>      -> Query AITalk Copilot (or type directly)")
    print("  /exit                -> Exit terminal")
    print("----------------------------------------------------------")

    active_lang = "en"
    ref_audio = None

    while True:
        try:
            prompt_str = f"chatterbox [{active_lang.upper()} | AITalk]> "
            line = input(prompt_str).strip()
            if not line:
                continue

            if line.lower() in ["/exit", "exit", "quit", ":q"]:
                print("Exiting Chatterbox Terminal. Goodbye!")
                break

            elif line.startswith("/languages"):
                list_languages()

            elif line.startswith("/lang"):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    new_lang = parts[1].strip().lower()
                    if new_lang in SUPPORTED_LANGUAGES:
                        active_lang = new_lang
                        print(f"[+] Active language switched to: {SUPPORTED_LANGUAGES[new_lang]} ({new_lang})")
                    else:
                        print(f"[!] Invalid language '{new_lang}'. Use /languages to see valid codes.")
                else:
                    print(f"Current language: {SUPPORTED_LANGUAGES.get(active_lang, active_lang)} ({active_lang})")

            elif line.startswith("/clone"):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    ref_audio = parts[1].strip()
                    print(f"[+] Reference audio set: {ref_audio}")
                else:
                    ref_audio = None
                    print("[+] Cleared reference audio.")

            elif line.startswith("/plans"):
                print("\n💳 Chatterbox Pricing & Scan-to-Pay Gateway:")
                print(f"   Official UPI Merchant ID: {os.getenv('UPI_ID', '9994152888-4#ybl')}")
                for p in billing.PLANS:
                    print(f"   - {p['name']:<18}: ₹{p['price_inr']} (${p['price_usd']}) -> {p['credits']:,} Credits")
                print()

            elif line.startswith("/synth"):
                text = line[6:].strip()
                if not text:
                    print("[!] Please provide text to synthesize: /synth <text>")
                    continue
                out_name = f"cli_speech_{active_lang}.wav"
                print(f"[*] Synthesizing in {SUPPORTED_LANGUAGES[active_lang]}...")
                sr, wav_np, _ = synthesize(
                    text=text,
                    language_id=active_lang,
                    audio_prompt_path=ref_audio
                )
                save_audio(wav_np, sr, out_name)
                print(f"[SUCCESS] Saved to {out_name} ({round(len(wav_np)/sr, 2)}s, {sr}Hz)")

            else:
                # Treat as AITalk Copilot Query
                query = line[4:].strip() if line.startswith("/ask") else line
                print(f"\n💬 [AITalk Copilot Thinking]...")
                res = copilot.ask_aitalk(query)
                print("\n" + res["answer"] + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox Multilingual TTS CLI - Synthesize speech & query AITalk copilot."
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to synthesize (max 300 characters)",
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="en",
        help="ISO 639-1 language code (e.g. en, fr, de, es, hi, zh). Default: en",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Path to save the generated audio WAV file. Default: output.wav",
    )
    parser.add_argument(
        "--ref-audio", "-r",
        type=str,
        default=None,
        help="Optional reference audio path or URL for voice cloning",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Speech expressiveness (0.25 - 2.0). Default: 0.5",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="CFG / pacing weight (0.0 - 1.0). Default: 0.5",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.1 - 2.0). Default: 0.8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default: 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device: 'mps', 'cuda', or 'cpu'. Default: auto-detect",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v3",
        choices=["v2", "v3"],
        help="Model version: 'v3' or 'v2'. Default: v3",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all supported language codes and exit",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive terminal session",
    )
    parser.add_argument(
        "--ai",
        type=str,
        nargs="?",
        const="interactive",
        help="Query the project-trained AITalk Copilot or enter AI terminal mode",
    )

    args = parser.parse_args()

    if args.list_languages:
        list_languages()
        sys.exit(0)

    if args.ai:
        if args.ai == "interactive":
            run_interactive_terminal(ai_mode=True)
            sys.exit(0)
        else:
            res = copilot.ask_aitalk(args.ai)
            print(res["answer"])
            sys.exit(0)

    if args.interactive or (not args.text and len(sys.argv) == 1):
        run_interactive_terminal(ai_mode=False)
        sys.exit(0)

    lang = args.lang.lower()
    if lang not in SUPPORTED_LANGUAGES:
        print(f"[ERROR] Unsupported language '{lang}'.")
        print(f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")
        sys.exit(1)

    print(f"[*] Target device: {get_target_device()}")
    print(f"[*] Language: {SUPPORTED_LANGUAGES[lang]} ({lang})")
    print(f"[*] Text: \"{args.text}\"")

    try:
        sr, wav_np, _ = synthesize(
            text=args.text,
            language_id=lang,
            audio_prompt_path=args.ref_audio,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            seed=args.seed,
            t3_model=args.model_version,
        )

        save_audio(wav_np, sr, args.output)
        duration = len(wav_np) / sr
        print(f"[+] Successfully saved {duration:.2f}s of audio to: {args.output}")

    except Exception as e:
        print(f"[ERROR] Synthesis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
