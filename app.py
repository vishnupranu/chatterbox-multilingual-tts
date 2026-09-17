"""
app.py
Interactive Gradio Web Interface for Chatterbox Multilingual TTS.
Supports 23 languages, zero-shot voice cloning with reference audio, and model parameter controls.
"""
import os
import gradio as gr
from tts_engine import (
    SUPPORTED_LANGUAGES,
    SAMPLE_CONFIG,
    synthesize,
    get_target_device,
)

DEVICE = get_target_device()
T3_MODEL = os.getenv("CHATTERBOX_MULTILINGUAL_T3_MODEL", "v3")

LANGUAGE_CHOICES = [
    (f"{name} ({code})", code) for code, name in sorted(SUPPORTED_LANGUAGES.items())
]


def on_language_change(lang_code):
    sample = SAMPLE_CONFIG.get(lang_code, {})
    default_text = sample.get("text", "")
    default_audio = sample.get("audio", None)
    return default_text, default_audio


def generate_tts_ui(
    text,
    language_code,
    ref_audio,
    exaggeration,
    cfg_weight,
    temperature,
    seed,
):
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    try:
        sr, wav_np, _ = synthesize(
            text=text.strip(),
            language_id=language_code,
            audio_prompt_path=ref_audio if ref_audio else None,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            seed=int(seed),
            t3_model=T3_MODEL,
        )
        return (sr, wav_np)
    except Exception as e:
        raise gr.Error(f"Synthesis failed: {str(e)}")


def create_ui():
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(title="Chatterbox Multilingual TTS", theme=custom_theme) as demo:
        gr.Markdown(
            f"""
            # 🎙️ Chatterbox Multilingual TTS
            **State-of-the-Art Open-Source Multilingual Text-to-Speech by Resemble AI**
            - **Model**: Chatterbox Multilingual {T3_MODEL.upper()}
            - **Device**: `{DEVICE}` (Apple Silicon MPS / CUDA / CPU)
            - **Languages**: 23 supported languages with zero-shot voice cloning
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                lang_dropdown = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value="en",
                    label="Language",
                    info="Choose language for synthesis",
                )

                text_input = gr.Textbox(
                    value=SAMPLE_CONFIG["en"]["text"],
                    label="Input Text",
                    placeholder="Enter up to 300 characters to synthesize...",
                    lines=4,
                    max_lines=6,
                )

                ref_audio_input = gr.Audio(
                    value=SAMPLE_CONFIG["en"]["audio"],
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Voice Reference Audio (Optional for Voice Cloning)",
                )

                with gr.Accordion("Advanced Voice Controls", open=False):
                    exaggeration_slider = gr.Slider(
                        minimum=0.25,
                        maximum=2.0,
                        step=0.05,
                        value=0.5,
                        label="Exaggeration / Expressiveness",
                        info="0.5 is neutral. Higher values produce more dramatic speech.",
                    )
                    cfg_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                        label="CFG / Pace Weight",
                        info="0.5 default. Set to 0.0 for cross-lingual accent transfer without reference accent.",
                    )
                    temp_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        step=0.05,
                        value=0.8,
                        label="Temperature",
                        info="Controls variation in generation.",
                    )
                    seed_input = gr.Number(
                        value=0,
                        precision=0,
                        label="Random Seed (0 for random)",
                    )

                generate_btn = gr.Button("⚡ Generate Speech", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 🔊 Generated Speech Output")
                audio_output = gr.Audio(label="Audio Output", autoplay=True)

        lang_dropdown.change(
            fn=on_language_change,
            inputs=[lang_dropdown],
            outputs=[text_input, ref_audio_input],
        )

        generate_btn.click(
            fn=generate_tts_ui,
            inputs=[
                text_input,
                lang_dropdown,
                ref_audio_input,
                exaggeration_slider,
                cfg_slider,
                temp_slider,
                seed_input,
            ],
            outputs=[audio_output],
        )

    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
