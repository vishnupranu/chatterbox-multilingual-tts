#!/usr/bin/env python3
"""
cli.py
Command-line interface for Chatterbox Multilingual TTS.
"""
import argparse
import sys
from tts_engine import (
    SUPPORTED_LANGUAGES,
    synthesize,
    save_audio,
    get_target_device,
)


def list_languages():
    print("\nAvailable Languages:")
    print("-" * 35)
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        print(f"  {code:<5} - {name}")
    print("-" * 35)


def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox Multilingual TTS CLI - Synthesize speech in 23 languages."
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
        help="CFG / pacing weight (0.0 - 1.0). Set to 0.0 for accent decoupling. Default: 0.5",
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
        help="Random seed for reproducibility. Default: 0 (random)",
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

    args = parser.parse_args()

    if args.list_languages:
        list_languages()
        sys.exit(0)

    if not args.text:
        parser.print_help()
        print("\nError: --text is required for synthesis.")
        sys.exit(1)

    lang = args.lang.lower()
    if lang not in SUPPORTED_LANGUAGES:
        print(f"Error: Unknown language code '{lang}'.")
        list_languages()
        sys.exit(1)

    device = args.device or get_target_device()
    print(f"Synthesizing [{lang}] on device '{device}'...")
    print(f"Text: \"{args.text}\"")

    try:
        sr, _, wav_tensor = synthesize(
            text=args.text,
            language_id=lang,
            audio_prompt_path=args.ref_audio,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            seed=args.seed,
            device=device,
            t3_model=args.model_version,
        )

        save_audio(args.output, wav_tensor, sr)
        print(f" Audio successfully saved to: {args.output} (sample rate: {sr} Hz)")

    except Exception as e:
        print(f"Synthesis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
