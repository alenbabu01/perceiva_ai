import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate speech audio using Google TTS. "
            "Supports: gTTS (Google Translate TTS, simple) and Google Cloud Text-to-Speech (higher quality)."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["gtts", "google-cloud"],
        default="gtts",
        help="TTS backend. 'gtts' is simple; 'google-cloud' supports WaveNet/Neural2 + SSML.",
    )
    parser.add_argument(
        "--text",
        default="Hello! This is a Google TTS sample.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--ssml",
        default=None,
        help=(
            "Optional SSML input (overrides --text). Example: "
            "'<speak>Hello <break time=\"300ms\"/> world</speak>'"
        ),
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code, e.g. en, en-us, hi, fr, de.",
    )
    parser.add_argument(
        "--out",
        default="google_tts.mp3",
        help="Output MP3 path.",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Speak more slowly.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help=(
            "(google-cloud) Optional voice name, e.g. 'en-US-Neural2-J' or 'en-US-Wavenet-D'. "
            "If omitted, Google selects a default for the language."
        ),
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="(google-cloud) Speaking rate. Typical range ~0.25 to 4.0.",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="(google-cloud) Pitch in semitones. Typical range ~-20.0 to 20.0.",
    )
    parser.add_argument(
        "--effects-profile-id",
        default=None,
        help=(
            "(google-cloud) Optional effects profile, e.g. 'handset-class-device' or 'small-bluetooth-speaker-class-device'."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["mp3", "wav"],
        default="mp3",
        help="(google-cloud) Output audio format.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "gtts":
        try:
            from gtts import gTTS
        except Exception as exc:
            print("Missing dependency: gTTS")
            print("Install with: pip install gTTS")
            print(f"Import error: {exc}")
            return 2

        if args.ssml:
            print("gTTS does not support SSML; use --backend google-cloud for SSML.")
            return 2

        tts = gTTS(text=args.text, lang=args.lang, slow=args.slow)
        tts.save(str(out_path))
        print(f"Wrote: {out_path.resolve()}")
        return 0

    # google-cloud backend
    try:
        from google.cloud import texttospeech
    except Exception as exc:
        print("Missing dependency: google-cloud-texttospeech")
        print("Install with: pip install google-cloud-texttospeech")
        print("Also requires Google Cloud credentials (Application Default Credentials).")
        print("Docs: https://cloud.google.com/text-to-speech/docs/before-you-begin")
        print(f"Import error: {exc}")
        return 2

    client = texttospeech.TextToSpeechClient()

    if args.ssml:
        synthesis_input = texttospeech.SynthesisInput(ssml=args.ssml)
    else:
        synthesis_input = texttospeech.SynthesisInput(text=args.text)

    voice_kwargs = {"language_code": args.lang}
    if args.voice:
        voice_kwargs["name"] = args.voice
    voice = texttospeech.VoiceSelectionParams(**voice_kwargs)

    effects_profile_id = [args.effects_profile_id] if args.effects_profile_id else None
    audio_config = texttospeech.AudioConfig(
        audio_encoding=(
            texttospeech.AudioEncoding.MP3
            if args.format == "mp3"
            else texttospeech.AudioEncoding.LINEAR16
        ),
        speaking_rate=args.speaking_rate,
        pitch=args.pitch,
        effects_profile_id=effects_profile_id,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    out_path.write_bytes(response.audio_content)
    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
