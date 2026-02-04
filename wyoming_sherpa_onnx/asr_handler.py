"""Wyoming ASR event handler for sherpa-onnx."""

import asyncio
import logging
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
        # Store audio chunks directly in memory (more efficient than WAV file)
        self._audio_chunks: list[bytes] = []
        self._sample_rate: int = 16000
        self._sample_width: int = 2
        self._channels: int = 1

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
            
            # Store format from first chunk
            if not self._audio_chunks:
                self._sample_rate = chunk.rate
                self._sample_width = chunk.width
                self._channels = chunk.channels

            self._audio_chunks.append(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped, starting transcription")

            if not self._audio_chunks:
                _LOGGER.warning("No audio received")
                await self.write_event(Transcript(text="").event())
                return False

            try:
                # Combine all chunks efficiently
                audio_bytes = b"".join(self._audio_chunks)
                total_samples = len(audio_bytes) // self._sample_width
                
                # Convert bytes to numpy array
                if self._sample_width == 2:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                elif self._sample_width == 4:
                    audio = np.frombuffer(audio_bytes, dtype=np.int32)
                else:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Free the bytes immediately
                del audio_bytes
                self._audio_chunks.clear()
                
                # Convert to float32
                if audio.dtype == np.int16:
                    waveform = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    waveform = audio.astype(np.float32) / 2147483648.0
                else:
                    waveform = audio.astype(np.float32)
                
                # Free the int audio array
                del audio

                # Make mono by averaging channels
                if self._channels > 1:
                    waveform = waveform.reshape(-1, self._channels).mean(axis=1)

            except Exception as e:
                _LOGGER.error("Failed to process audio: %s", e)
                self._audio_chunks.clear()
                await self.write_event(Transcript(text=f"ERROR: {e}").event())
                return False

            # Transcribe
            async with self.model_lock:
                try:
                    text = self.engine.recognize(
                        waveform,
                        self._sample_rate,
                        language=self.request_language,
                    )
                    _LOGGER.info("Transcribed: %s", text[:100] if text else "(empty)")

                except Exception as e:
                    _LOGGER.exception("Transcription failed")
                    await self.write_event(Transcript(text=f"ERROR: {e}").event())
                    return False
                finally:
                    # Free waveform after use
                    del waveform

            await self.write_event(Transcript(text=text).event())

            # Reset for next request
            self.request_language = None

            return False

        _LOGGER.warning("Unhandled event type: %s", event.type)
        return True
