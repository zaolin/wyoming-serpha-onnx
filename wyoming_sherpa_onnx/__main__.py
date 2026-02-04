#!/usr/bin/env python3
"""Wyoming server for sherpa-onnx ASR and TTS."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from functools import partial
from pathlib import Path
from typing import Optional

from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import SERVICE_NAME, __version__
from .asr_handler import SherpaASREventHandler
from .engine import (
    ASRConfig, SherpaASREngine, SherpaTTSEngine, TTSConfig,
    get_gpu_memory_info, estimate_model_size,
)
from .tts_handler import SherpaTTSEventHandler

_LOGGER = logging.getLogger(__name__)

# Language codes supported by SenseVoice / multilingual models
_LANGUAGE_CODES = (
    "de", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh",
    "ar", "cs", "da", "el", "fi", "he", "hi", "hu", "id", "no", "ro", "sk",
    "sv", "th", "tr", "uk", "vi", "yue",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Wyoming sherpa-onnx ASR/TTS server")

    # Server URIs
    parser.add_argument(
        "--asr-uri",
        default=os.environ.get("ASR_URI", "tcp://0.0.0.0:10300"),
        help="ASR server URI (default: tcp://0.0.0.0:10300)",
    )
    parser.add_argument(
        "--tts-uri",
        default=os.environ.get("TTS_URI", "tcp://0.0.0.0:10400"),
        help="TTS server URI (default: tcp://0.0.0.0:10400)",
    )

    # Model paths
    parser.add_argument(
        "--asr-model",
        type=Path,
        default=Path(os.environ.get("ASR_MODEL_PATH", "/app/models/asr")),
        help="Path to ASR model directory",
    )
    parser.add_argument(
        "--tts-model",
        type=Path,
        default=Path(os.environ.get("TTS_MODEL_PATH", "/app/models/tts")),
        help="Path to TTS model directory",
    )
    parser.add_argument(
        "--asr-mode",
        choices=["auto", "offline", "online"],
        default=os.environ.get("ASR_MODE", "auto"),
        help="ASR mode (auto, offline, online). Detects automatically by default.",
    )
    parser.add_argument(
        "--asr-languages",
        type=str,
        default=os.environ.get("ASR_LANGUAGES", ""),
        help="Override ASR languages (comma-separated, e.g., 'de,en')",
    )
    parser.add_argument(
        "--tts-languages",
        type=str,
        default=os.environ.get("TTS_LANGUAGES", ""),
        help="Override TTS languages (comma-separated, e.g., 'de')",
    )

    # Options
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=int(os.environ.get("TTS_SPEAKER_ID", "0")),
        help="Default TTS speaker ID",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=os.environ.get("USE_GPU", "true").lower() == "true",
        help="Use GPU acceleration for both ASR and TTS (overridden by --asr-gpu/--tts-gpu)",
    )
    parser.add_argument(
        "--asr-gpu",
        type=lambda x: x.lower() == "true",
        default=os.environ.get("ASR_GPU"),  # None if not set
        help="Use GPU for ASR (true/false, overrides --use-gpu for ASR)",
    )
    parser.add_argument(
        "--tts-gpu",
        type=lambda x: x.lower() == "true",
        default=os.environ.get("TTS_GPU"),  # None if not set
        help="Use GPU for TTS (true/false, overrides --use-gpu for TTS)",
    )
    parser.add_argument(
        "--asr-only",
        action="store_true",
        default=os.environ.get("ASR_ONLY", "false").lower() == "true",
        help="Run only ASR server",
    )
    parser.add_argument(
        "--tts-only",
        action="store_true",
        default=os.environ.get("TTS_ONLY", "false").lower() == "true",
        help="Run only TTS server",
    )

    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    return parser.parse_args()


# Service name constants for Wyoming info
SERVICE_NAME_ASR = "Sherpa ASR"
SERVICE_NAME_TTS = "Sherpa TTS"


def extract_voice_name(model_name: str) -> str:
    """
    Extract a friendly voice name from model path.
    
    Examples:
    - vits-piper-de_DE-thorsten-high -> Thorsten (high)
    - kokoro-v1.0 -> Kokoro v1.0
    - whisper-large-v3 -> Whisper Large V3
    """
    import re
    
    # Remove common prefixes
    name = model_name
    for prefix in ["vits-piper-", "vits-", "piper-", "sherpa-onnx-"]:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    
    # Extract quality variant (high, medium, low) if present
    quality = ""
    for q in ["high", "medium", "low", "hq", "lq"]:
        if name.lower().endswith(f"-{q}"):
            quality = q
            name = name[:-len(q)-1]
            break
    
    # Remove language prefix like de_DE- or en-US-
    name = re.sub(r'^[a-z]{2}[_-][A-Z]{2}[_-]', '', name)
    
    # Title case the remaining name
    voice_name = name.replace("-", " ").replace("_", " ").title()
    
    # Add quality variant if present
    if quality:
        voice_name = f"{voice_name} ({quality})"
    
    return voice_name


def detect_languages_from_model(model_name: str, model_type: str) -> list[str]:
    """
    Detect supported languages from model name.
    
    Returns list of language codes.
    """
    name_lower = model_name.lower()
    
    # Multilingual model types (always multilingual regardless of name)
    multilingual_types = ["whisper", "sensevoice"]
    if model_type.lower() in multilingual_types:
        _LOGGER.debug("Model type '%s' is inherently multilingual", model_type)
        return list(_LANGUAGE_CODES)
    
    # Multilingual patterns in model name
    multilingual_patterns = [
        "whisper",  # Whisper supports 99 languages
        "sense-voice", "sensevoice",  # SenseVoice multilingual
        "multi-lang", "multilingual", "multi_lang",
        "zh-en", "en-zh", "bilingual",
    ]
    
    if any(p in name_lower for p in multilingual_patterns):
        # Return common language codes for multilingual models
        return list(_LANGUAGE_CODES)
    
    # Language code patterns in model names (ISO format)
    # Match patterns like: de_DE, en_US, zh_CN, fr-FR, etc.
    import re
    
    lang_codes = set()
    
    # Pattern: xx_XX or xx-XX anywhere in name (e.g., de_DE, en-US, zh_CN)
    # This matches patterns like "piper-de_DE-thorsten" -> "de"
    matches = re.findall(r'([a-z]{2})[_-]([A-Z]{2})', model_name)
    for lang, region in matches:
        lang_codes.add(lang.lower())
        _LOGGER.debug("Detected language from region code: %s_%s -> %s", lang, region, lang)
    
    # Pattern: standalone language codes between separators
    # Match -de- or _en_ patterns in lowercase model name
    matches = re.findall(r'[_-]([a-z]{2})[_-]', name_lower)
    for lang in matches:
        if lang in ("de", "en", "fr", "es", "it", "zh", "ja", "ko", "ru", 
                    "pl", "nl", "pt", "sv", "da", "no", "fi", "cs", "sk",
                    "hu", "ro", "el", "tr", "ar", "he", "hi", "id", "th", "vi", "uk", "yue"):
            lang_codes.add(lang)
            _LOGGER.debug("Detected standalone language code: %s", lang)
    
    # Explicit language words
    lang_map = {
        "german": "de", "deutsch": "de",
        "english": "en",
        "french": "fr", "francais": "fr",
        "spanish": "es", "espanol": "es",
        "italian": "it", "italiano": "it",
        "chinese": "zh", "mandarin": "zh",
        "japanese": "ja",
        "korean": "ko",
        "russian": "ru",
        "polish": "pl",
        "dutch": "nl",
        "portuguese": "pt",
        "swedish": "sv",
        "danish": "da",
        "norwegian": "no",
        "finnish": "fi",
        "czech": "cs",
        "slovak": "sk",
        "hungarian": "hu",
        "romanian": "ro",
        "greek": "el",
        "turkish": "tr",
        "arabic": "ar",
        "hebrew": "he",
        "hindi": "hi",
        "indonesian": "id",
        "thai": "th",
        "vietnamese": "vi",
        "ukrainian": "uk",
        "cantonese": "yue",
    }
    
    for word, code in lang_map.items():
        if word in name_lower:
            lang_codes.add(code)
    
    # If no language detected, check model type defaults
    if not lang_codes:
        if model_type in ("moonshine",):
            # Moonshine is English only
            lang_codes.add("en")
        elif model_type in ("paraformer",):
            # Paraformer is Chinese focused
            lang_codes.add("zh")
        else:
            # Default to English
            lang_codes.add("en")
            _LOGGER.warning("No language detected in model name '%s' (type=%s), defaulting to English", model_name, model_type)
    
    result = sorted(lang_codes)
    _LOGGER.info("Language detection for '%s' (%s): %s", model_name, model_type, result)
    return result


def build_asr_info(engine: "SherpaASREngine", language_override: str = "") -> Info:
    """Build Wyoming info for ASR service with detected model info."""
    model_type = engine.model_type or "unknown"
    model_name = engine.model_name
    
    # Use language override if provided, otherwise detect from model
    if language_override:
        languages = [l.strip() for l in language_override.split(",") if l.strip()]
        _LOGGER.info("Using ASR language override: %s", languages)
    else:
        languages = detect_languages_from_model(model_name, model_type)
    is_multilingual = len(languages) > 5
    
    descriptions = {
        "whisper": f"Whisper {'multilingual' if is_multilingual else ''} ASR",
        "sensevoice": "SenseVoice multilingual ASR (99 languages)",
        "moonshine": "Moonshine fast ASR (English)",
        "transducer": "Zipformer/Conformer transducer ASR",
        "paraformer": "Paraformer ASR (Chinese)",
        "ctc": "CTC-based ASR",
    }
    description = descriptions.get(model_type, f"{model_type} ASR")
    
    # Use clean constant name for HA (this is the "engine" or "program" name)
    program_name = SERVICE_NAME_ASR
    
    # Friendly model name for display
    friendly_model_name = extract_voice_name(model_name) if model_name else model_type.capitalize()
    
    info = Info(
        asr=[
            AsrProgram(
                name=program_name,
                description=f"Sherpa-ONNX: {description}",
                attribution=Attribution(
                    name="k2-fsa",
                    url="https://github.com/k2-fsa/sherpa-onnx",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=f"{model_type.capitalize()} model ({len(languages)} languages)",
                        attribution=Attribution(
                            name="sherpa-onnx",
                            url="https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html",
                        ),
                        installed=True,
                        languages=languages,
                        version="1.0",
                    )
                ],
            )
        ],
    )
    _LOGGER.info("ASR Info: program='%s', model='%s', languages=%s", program_name, model_name, languages)
    return info


def build_tts_info(engine: "SherpaTTSEngine", speaker_id: int, language_override: str = "") -> Info:
    """Build Wyoming info for TTS service with detected model info."""
    model_type = engine.model_type or "unknown"
    model_name = engine.model_name
    
    # Use language override if provided, otherwise detect from model
    if language_override:
        languages = [l.strip() for l in language_override.split(",") if l.strip()]
        _LOGGER.info("Using TTS language override: %s", languages)
    else:
        languages = detect_languages_from_model(model_name, model_type)
    is_multilingual = len(languages) > 1
    
    # Determine description based on model type
    descriptions = {
        "vits": "VITS/Piper neural TTS",
        "matcha": "Matcha neural TTS",
        "kokoro": "Kokoro multi-speaker TTS",
    }
    description = descriptions.get(model_type, f"{model_type} TTS")
    
    lang_desc = ", ".join(languages[:3])
    if len(languages) > 3:
        lang_desc += f" +{len(languages) - 3} more"
    
    # Use clean constant name for HA (this is the "engine" or "program" name)
    program_name = SERVICE_NAME_TTS
    
    # Friendly voice name for display (e.g., "Thorsten (high)")
    voice_display_name = extract_voice_name(model_name) if model_name else model_type.capitalize()
    
    info = Info(
        tts=[
            TtsProgram(
                name=program_name,
                description=f"Sherpa-ONNX: {description}",
                attribution=Attribution(
                    name="k2-fsa",
                    url="https://github.com/k2-fsa/sherpa-onnx",
                ),
                installed=True,
                version=__version__,
                voices=[
                    TtsVoice(
                        name=model_name,  # Keep full name as ID for Wyoming protocol
                        description=voice_display_name,  # Friendly name shown in HA
                        attribution=Attribution(
                            name="sherpa-onnx",
                            url="https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html",
                        ),
                        installed=True,
                        languages=languages,
                        version="1.0",
                    )
                ],
            )
        ],
    )
    _LOGGER.info("TTS Info: program='%s', voice='%s' (%s), languages=%s", program_name, voice_display_name, model_name, languages)
    return info


async def run_asr_server(
    uri: str,
    engine: SherpaASREngine,
    model_lock: asyncio.Lock,
    language_override: str = "",
) -> None:
    """Run ASR server."""
    wyoming_info = build_asr_info(engine, language_override)
    server = AsyncServer.from_uri(uri)

    _LOGGER.info("ASR server listening on %s", uri)

    await server.run(
        partial(
            SherpaASREventHandler,
            wyoming_info,
            engine,
            model_lock,
        )
    )


async def run_tts_server(
    uri: str,
    engine: SherpaTTSEngine,
    model_lock: asyncio.Lock,
    speaker_id: int,
    language_override: str = "",
) -> None:
    """Run TTS server."""
    wyoming_info = build_tts_info(engine, speaker_id, language_override)
    server = AsyncServer.from_uri(uri)

    _LOGGER.info("TTS server listening on %s", uri)

    await server.run(
        partial(
            SherpaTTSEventHandler,
            wyoming_info,
            engine,
            model_lock,
        )
    )


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    _LOGGER.info("Starting %s v%s", SERVICE_NAME, __version__)

    # Shared locks for model access
    asr_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()

    # Shutdown handling
    shutdown_event = asyncio.Event()

    def handle_signal() -> None:
        _LOGGER.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    tasks = []

    # Determine GPU usage for each model
    # Priority: --asr-gpu/--tts-gpu > ASR_GPU/TTS_GPU env > --use-gpu
    def parse_gpu_flag(arg_value, env_name, default):
        if arg_value is not None:
            # CLI argument was provided
            if isinstance(arg_value, str):
                return arg_value.lower() == "true"
            return bool(arg_value)
        env_val = os.environ.get(env_name)
        if env_val is not None:
            return env_val.lower() == "true"
        return default
    
    asr_use_gpu = parse_gpu_flag(args.asr_gpu, "ASR_GPU", args.use_gpu)
    tts_use_gpu = parse_gpu_flag(args.tts_gpu, "TTS_GPU", args.use_gpu)
    
    # Check GPU memory for each model that will use GPU
    asr_path = args.asr_model if not args.tts_only and args.asr_model.exists() else None
    tts_path = args.tts_model if not args.asr_only and args.tts_model.exists() else None
    
    if asr_use_gpu and asr_path:
        total, used, free = get_gpu_memory_info()
        asr_size = estimate_model_size(asr_path)
        if total > 0 and asr_size + 300 > free:
            _LOGGER.warning(
                "ASR model (~%d MB) may not fit in GPU memory (%d MB free), using CPU",
                asr_size, free
            )
            asr_use_gpu = False
        else:
            _LOGGER.info("ASR will use GPU (model ~%d MB, %d MB free)", asr_size, free)
    
    if tts_use_gpu and tts_path:
        total, used, free = get_gpu_memory_info()
        tts_size = estimate_model_size(tts_path)
        # Account for ASR if also using GPU
        reserved = estimate_model_size(asr_path) if asr_use_gpu and asr_path else 0
        if total > 0 and tts_size + reserved + 300 > free:
            _LOGGER.warning(
                "TTS model (~%d MB) may not fit in GPU memory (%d MB free, %d MB reserved for ASR), using CPU",
                tts_size, free, reserved
            )
            tts_use_gpu = False
        else:
            _LOGGER.info("TTS will use GPU (model ~%d MB)", tts_size)

    # Initialize ASR
    asr_engine: Optional[SherpaASREngine] = None
    if not args.tts_only and args.asr_model.exists():
        # Use first configured language for engine if specified
        engine_lang = ""
        if args.asr_languages:
            engine_lang = args.asr_languages.split(",")[0].strip()

        asr_config = ASRConfig(
            model_path=args.asr_model,
            use_gpu=asr_use_gpu,
            provider="cuda" if asr_use_gpu else "cpu",
            language=engine_lang,
            mode=args.asr_mode,
        )
        asr_engine = SherpaASREngine(asr_config)
        await asr_engine.load()
        _LOGGER.info("ASR engine loaded (%s)", "GPU" if asr_use_gpu else "CPU")

        tasks.append(
            asyncio.create_task(
                run_asr_server(args.asr_uri, asr_engine, asr_lock, args.asr_languages)
            )
        )
    elif not args.tts_only:
        _LOGGER.warning("ASR model not found at %s", args.asr_model)

    # Initialize TTS
    tts_engine: Optional[SherpaTTSEngine] = None
    if not args.asr_only and args.tts_model.exists():
        tts_config = TTSConfig(
            model_path=args.tts_model,
            use_gpu=tts_use_gpu,
            provider="cuda" if tts_use_gpu else "cpu",
            speaker_id=args.speaker_id,
        )
        tts_engine = SherpaTTSEngine(tts_config)
        await tts_engine.load()
        _LOGGER.info("TTS engine loaded")

        tasks.append(
            asyncio.create_task(
                run_tts_server(args.tts_uri, tts_engine, tts_lock, args.speaker_id, args.tts_languages)
            )
        )
    elif not args.asr_only:
        _LOGGER.warning("TTS model not found at %s", args.tts_model)

    if not tasks:
        _LOGGER.error("No models found. Please provide ASR and/or TTS models.")
        sys.exit(1)

    _LOGGER.info("Ready")

    # Wait for shutdown
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    done, pending = await asyncio.wait(
        tasks + [shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel remaining tasks
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _LOGGER.info("Shutdown complete")


def run() -> None:
    """Entry point for console script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
