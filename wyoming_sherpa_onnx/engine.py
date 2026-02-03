"""Sherpa-onnx engine wrapper for ASR and TTS."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Audio constants
ASR_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1


@dataclass
class ASRConfig:
    """Configuration for ASR engine."""

    model_path: Path
    use_gpu: bool = True
    num_threads: int = 4
    provider: str = "cuda"  # cuda, cpu


@dataclass
class TTSConfig:
    """Configuration for TTS engine."""

    model_path: Path
    tokens_path: Optional[Path] = None
    data_dir: Optional[Path] = None
    lexicon_path: Optional[Path] = None
    dict_dir: Optional[Path] = None
    use_gpu: bool = True
    num_threads: int = 4
    provider: str = "cuda"
    speaker_id: int = 0
    speed: float = 1.0


class SherpaASREngine:
    """Sherpa-onnx ASR engine wrapper."""

    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self._recognizer = None

    async def load(self) -> None:
        """Load the ASR model."""
        import sherpa_onnx

        _LOGGER.info("Loading ASR model from %s", self.config.model_path)

        # Detect model type based on files present
        model_path = self.config.model_path

        # Check for SenseVoice model
        sense_voice_model = model_path / "model.onnx"
        if sense_voice_model.exists():
            _LOGGER.info("Detected SenseVoice model")
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(sense_voice_model),
                tokens=str(model_path / "tokens.txt"),
                use_itn=True,
                num_threads=self.config.num_threads,
                provider=self.config.provider if self.config.use_gpu else "cpu",
            )
            return

        # Check for Whisper model
        encoder = model_path / "encoder.onnx"
        decoder = model_path / "decoder.onnx"
        if encoder.exists() and decoder.exists():
            _LOGGER.info("Detected Whisper model")
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(encoder),
                decoder=str(decoder),
                tokens=str(model_path / "tokens.txt"),
                num_threads=self.config.num_threads,
                provider=self.config.provider if self.config.use_gpu else "cpu",
            )
            return

        # Check for transducer model (streaming)
        transducer_encoder = model_path / "encoder.onnx"
        transducer_decoder = model_path / "decoder.onnx"
        transducer_joiner = model_path / "joiner.onnx"
        if (
            transducer_encoder.exists()
            and transducer_decoder.exists()
            and transducer_joiner.exists()
        ):
            _LOGGER.info("Detected Transducer model")
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(transducer_encoder),
                decoder=str(transducer_decoder),
                joiner=str(transducer_joiner),
                tokens=str(model_path / "tokens.txt"),
                num_threads=self.config.num_threads,
                provider=self.config.provider if self.config.use_gpu else "cpu",
            )
            return

        raise ValueError(f"Could not detect model type in {model_path}")

    def recognize(
        self, audio: np.ndarray, sample_rate: int, language: Optional[str] = None
    ) -> str:
        """Recognize speech from audio samples."""
        if self._recognizer is None:
            raise RuntimeError("ASR engine not loaded")

        # Resample if needed
        if sample_rate != ASR_SAMPLE_RATE:
            import scipy.signal

            audio = scipy.signal.resample(
                audio, int(len(audio) * ASR_SAMPLE_RATE / sample_rate)
            )
            sample_rate = ASR_SAMPLE_RATE

        # Ensure float32
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        # Create stream and decode
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)

        return stream.result.text.strip()


class SherpaTTSEngine:
    """Sherpa-onnx TTS engine wrapper."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._tts = None
        self.sample_rate = TTS_SAMPLE_RATE

    async def load(self) -> None:
        """Load the TTS model."""
        import sherpa_onnx

        _LOGGER.info("Loading TTS model from %s", self.config.model_path)

        model_path = self.config.model_path

        # Look for VITS model file
        vits_model = model_path / "model.onnx"
        if not vits_model.exists():
            # Try alternate name
            vits_model = model_path / "en_US-lessac-medium.onnx"

        # Find any .onnx file
        if not vits_model.exists():
            onnx_files = list(model_path.glob("*.onnx"))
            if onnx_files:
                vits_model = onnx_files[0]
            else:
                raise ValueError(f"No ONNX model found in {model_path}")

        # Look for tokens file
        tokens_file = model_path / "tokens.txt"
        if not tokens_file.exists():
            raise ValueError(f"tokens.txt not found in {model_path}")

        # Look for data directory (for Piper models)
        data_dir = model_path / "espeak-ng-data"
        lexicon = model_path / "lexicon.txt"
        dict_dir = model_path / "dict"

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(vits_model),
                    tokens=str(tokens_file),
                    data_dir=str(data_dir) if data_dir.exists() else "",
                    lexicon=str(lexicon) if lexicon.exists() else "",
                    dict_dir=str(dict_dir) if dict_dir.exists() else "",
                ),
                provider=self.config.provider if self.config.use_gpu else "cpu",
                num_threads=self.config.num_threads,
            ),
            max_num_sentences=1,
        )

        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self.sample_rate = self._tts.sample_rate
        _LOGGER.info("TTS loaded, sample rate: %d", self.sample_rate)

    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> np.ndarray:
        """Synthesize speech from text."""
        if self._tts is None:
            raise RuntimeError("TTS engine not loaded")

        sid = speaker_id if speaker_id is not None else self.config.speaker_id

        audio = self._tts.generate(
            text,
            sid=sid,
            speed=self.config.speed,
        )

        return audio.samples
