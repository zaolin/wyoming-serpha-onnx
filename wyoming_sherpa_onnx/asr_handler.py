"""Wyoming ASR event handler for sherpa-onnx."""

import asyncio
import logging
import os
import tempfile
import wave
from typing import Any, Optional

import numpy as np

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .engine import SherpaASREngine

_LOGGER = logging.getLogger(__name__)


class SherpaASREventHandler(AsyncEventHandler):
    """Event handler for ASR clients."""

    def __init__(
        self,
        wyoming_info: Info,
        engine: SherpaASREngine,
        model_lock: asyncio.Lock,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info = wyoming_info
        self.wyoming_info_event = wyoming_info.event()
        self.engine = engine
        self.model_lock = model_lock

        self.request_language: Optional[str] = None
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming events."""

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent ASR info")
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.request_language = transcribe.language
            _LOGGER.debug("Language requested: %s", self.request_language)
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped, starting transcription")

            if self._wav_file is None:
                _LOGGER.warning("No audio received")
                await self.write_event(Transcript(text="").event())
                return False

            self._wav_file.close()
            self._wav_file = None

            # Read audio file
            try:
                import soundfile as sf

                waveform, sample_rate = sf.read(self._wav_path, dtype="float32")

                # Make mono by averaging channels
                if len(waveform.shape) > 1:
                    waveform = np.mean(waveform, axis=1)

            except Exception as e:
                _LOGGER.error("Failed to read audio: %s", e)
                await self.write_event(Transcript(text=f"ERROR: {e}").event())
                return False

            # Transcribe
            async with self.model_lock:
                try:
                    text = self.engine.recognize(
                        waveform,
                        sample_rate,
                        language=self.request_language,
                    )
                    _LOGGER.info("Transcribed: %s", text[:100] if text else "(empty)")

                except Exception as e:
                    _LOGGER.exception("Transcription failed")
                    await self.write_event(Transcript(text=f"ERROR: {e}").event())
                    return False

            await self.write_event(Transcript(text=text).event())

            # Reset for next request
            self.request_language = None

            return False

        _LOGGER.warning("Unhandled event type: %s", event.type)
        return True
