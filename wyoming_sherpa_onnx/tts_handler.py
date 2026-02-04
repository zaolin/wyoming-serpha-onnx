"""Wyoming TTS event handler for sherpa-onnx."""

import asyncio
import logging
from typing import Any, Optional

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

from .engine import SAMPLE_WIDTH, CHANNELS, SherpaTTSEngine

_LOGGER = logging.getLogger(__name__)


class SherpaTTSEventHandler(AsyncEventHandler):
    """Event handler for TTS clients."""

    def __init__(
        self,
        wyoming_info: Info,
        engine: SherpaTTSEngine,
        model_lock: asyncio.Lock,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info = wyoming_info
        self.wyoming_info_event = wyoming_info.event()
        self.engine = engine
        self.model_lock = model_lock

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming events."""

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent TTS info")
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            await self._handle_synthesize(synthesize)
            return False

        _LOGGER.warning("Unhandled event type: %s", event.type)
        return True

    async def _handle_synthesize(self, synthesize: Synthesize) -> None:
        """Process synthesis request."""
        text = synthesize.text.strip()

        if not text:
            _LOGGER.warning("Empty text, skipping synthesis")
            return

        # Get speaker ID from voice name if specified
        speaker_id: Optional[int] = None
        if synthesize.voice and synthesize.voice.name:
            try:
                speaker_id = int(synthesize.voice.name)
            except ValueError:
                _LOGGER.debug(
                    "Voice name '%s' is not a speaker ID, using default",
                    synthesize.voice.name,
                )

        _LOGGER.debug(
            "Synthesizing: %r (speaker_id=%s)",
            text[:50] if len(text) > 50 else text,
            speaker_id,
        )

        # Synthesize and convert audio inside lock to avoid memory issues
        async with self.model_lock:
            try:
                audio_samples = self.engine.synthesize(text, speaker_id=speaker_id)
                
                # Ensure we have a numpy array (sherpa-onnx may return different types)
                if not isinstance(audio_samples, np.ndarray):
                    audio_samples = np.array(audio_samples, dtype=np.float32)
                
                # Convert float32 samples to int16 bytes
                # Clip to prevent overflow, scale, and convert
                audio_clipped = np.clip(audio_samples, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                sample_rate = self.engine.sample_rate
                num_samples = len(audio_int16)
                
                # Free arrays
                del audio_samples, audio_clipped, audio_int16
                
            except Exception as e:
                _LOGGER.exception("Synthesis failed: %s", e)
                return

        # Send audio (outside lock for better concurrency)
        await self.write_event(
            AudioStart(
                rate=sample_rate,
                width=SAMPLE_WIDTH,
                channels=CHANNELS,
            ).event()
        )

        # Send in chunks (4096 samples per chunk)
        chunk_size = 4096 * SAMPLE_WIDTH
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await self.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=sample_rate,
                    width=SAMPLE_WIDTH,
                    channels=CHANNELS,
                ).event()
            )

        await self.write_event(AudioStop().event())

        _LOGGER.info(
            "Synthesized %d chars -> %d samples",
            len(text),
            num_samples,
        )
