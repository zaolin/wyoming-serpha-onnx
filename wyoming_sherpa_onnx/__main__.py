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
from .engine import ASRConfig, SherpaASREngine, SherpaTTSEngine, TTSConfig
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
        help="Use GPU acceleration",
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


def build_asr_info() -> Info:
    """Build Wyoming info for ASR service."""
    return Info(
        asr=[
            AsrProgram(
                name=SERVICE_NAME,
                description="Sherpa-ONNX speech recognition",
                attribution=Attribution(
                    name="k2-fsa",
                    url="https://github.com/k2-fsa/sherpa-onnx",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="sensevoice",
                        description="Multilingual ASR (99 languages)",
                        attribution=Attribution(
                            name="FunAudioLLM",
                            url="https://github.com/FunAudioLLM/SenseVoice",
                        ),
                        installed=True,
                        languages=list(_LANGUAGE_CODES),
                        version="1.0",
                    )
                ],
            )
        ],
    )


def build_tts_info(speaker_id: int) -> Info:
    """Build Wyoming info for TTS service."""
    return Info(
        tts=[
            TtsProgram(
                name=SERVICE_NAME,
                description="Sherpa-ONNX text-to-speech",
                attribution=Attribution(
                    name="k2-fsa",
                    url="https://github.com/k2-fsa/sherpa-onnx",
                ),
                installed=True,
                version=__version__,
                voices=[
                    TtsVoice(
                        name=str(speaker_id),
                        description="Default voice",
                        attribution=Attribution(name="Piper", url="https://github.com/rhasspy/piper"),
                        installed=True,
                        languages=["de"],
                        version="1.0",
                    )
                ],
            )
        ],
    )


async def run_asr_server(
    uri: str,
    engine: SherpaASREngine,
    model_lock: asyncio.Lock,
) -> None:
    """Run ASR server."""
    wyoming_info = build_asr_info()
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
) -> None:
    """Run TTS server."""
    wyoming_info = build_tts_info(speaker_id)
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

    # Initialize ASR
    asr_engine: Optional[SherpaASREngine] = None
    if not args.tts_only and args.asr_model.exists():
        asr_config = ASRConfig(
            model_path=args.asr_model,
            use_gpu=args.use_gpu,
            provider="cuda" if args.use_gpu else "cpu",
        )
        asr_engine = SherpaASREngine(asr_config)
        await asr_engine.load()
        _LOGGER.info("ASR engine loaded")

        tasks.append(
            asyncio.create_task(
                run_asr_server(args.asr_uri, asr_engine, asr_lock)
            )
        )
    elif not args.tts_only:
        _LOGGER.warning("ASR model not found at %s", args.asr_model)

    # Initialize TTS
    tts_engine: Optional[SherpaTTSEngine] = None
    if not args.asr_only and args.tts_model.exists():
        tts_config = TTSConfig(
            model_path=args.tts_model,
            use_gpu=args.use_gpu,
            provider="cuda" if args.use_gpu else "cpu",
            speaker_id=args.speaker_id,
        )
        tts_engine = SherpaTTSEngine(tts_config)
        await tts_engine.load()
        _LOGGER.info("TTS engine loaded")

        tasks.append(
            asyncio.create_task(
                run_tts_server(args.tts_uri, tts_engine, tts_lock, args.speaker_id)
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
