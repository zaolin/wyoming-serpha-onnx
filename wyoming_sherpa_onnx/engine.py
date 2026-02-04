"""Sherpa-onnx engine wrapper for ASR and TTS."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Audio constants
ASR_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1


def get_gpu_memory_info() -> Tuple[int, int, int]:
    """
    Get available memory for GPU inference on Jetson.
    
    Jetson uses unified memory architecture where CPU and GPU share RAM.
    Reports system memory as GPU memory.
    
    Returns:
        Tuple of (total_mb, used_mb, free_mb)
    """
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = int(parts[1]) // 1024  # Convert KB to MB
                    meminfo[key] = value
            
            total = meminfo.get("MemTotal", 0)
            free = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            used = total - free
            
            _LOGGER.debug("Jetson unified memory: total=%d MB, free=%d MB", total, free)
            return total, used, free
    except Exception as e:
        _LOGGER.warning("Could not read /proc/meminfo: %s", e)
    
    return 0, 0, 0


def estimate_model_size(model_path: Path) -> int:
    """
    Estimate model size in MB from ONNX files.
    
    Returns:
        Estimated size in MB
    """
    total_bytes = 0
    
    # Sum all ONNX files
    for onnx_file in model_path.glob("**/*.onnx"):
        total_bytes += onnx_file.stat().st_size
    
    # Also count .weights files (used by Whisper)
    for weights_file in model_path.glob("**/*.weights"):
        total_bytes += weights_file.stat().st_size
    
    # Also count .bin files (Kokoro voices, etc)
    for bin_file in model_path.glob("**/*.bin"):
        total_bytes += bin_file.stat().st_size
    
    # Convert to MB and add overhead (~20% for GPU buffers)
    size_mb = int((total_bytes / 1024 / 1024) * 1.2)
    return size_mb


def check_gpu_memory_for_models(
    asr_path: Optional[Path], 
    tts_path: Optional[Path],
    min_free_mb: int = 500,
) -> Tuple[bool, str]:
    """
    Check if there's enough GPU memory for the models.
    
    Returns:
        Tuple of (ok, message)
    """
    total, used, free = get_gpu_memory_info()
    
    if total == 0:
        return True, "GPU memory info not available, proceeding anyway"
    
    asr_size = estimate_model_size(asr_path) if asr_path and asr_path.exists() else 0
    tts_size = estimate_model_size(tts_path) if tts_path and tts_path.exists() else 0
    
    required = asr_size + tts_size + min_free_mb
    
    _LOGGER.info(
        "GPU memory: %d MB total, %d MB used, %d MB free",
        total, used, free,
    )
    _LOGGER.info(
        "Model sizes: ASR=%d MB, TTS=%d MB, required=%d MB (incl. %d MB buffer)",
        asr_size, tts_size, required, min_free_mb,
    )
    
    if required > free:
        return False, (
            f"Insufficient GPU memory: need ~{required} MB but only {free} MB free. "
            f"Consider using smaller models or --use-gpu=false"
        )
    
    return True, f"GPU memory OK: {free} MB free, ~{required} MB required"


@dataclass
class ASRConfig:
    """Configuration for ASR engine."""
    model_path: Path
    use_gpu: bool = True
    num_threads: int = 4
    provider: str = "cuda"  # cuda, cpu
    language: str = ""  # Language code for Whisper (empty = auto-detect)
    mode: str = "auto"  # auto, offline, online


class SherpaASREngine:
    """ASR engine using sherpa-onnx."""

    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self._recognizer = None
        self._model_type = None
        self._model_name = ""

    def _detect_model_type(self, model_path: Path) -> tuple[str, dict]:
        """
        Detect model type and return type string + config dict.
        
        Supported types:
        - whisper
        - sensevoice
        - moonshine
        - transducer (zipformer/conformer)
        - paraformer
        - ctc (nemo/zipformer)
        """
        tokens = _find_file(model_path, ["tokens.txt", "*.txt"])
        
        # Check for .model_name file (created by download script)
        model_name_file = model_path / ".model_name"
        original_name = ""
        if model_name_file.exists():
            try:
                original_name = model_name_file.read_text().strip().lower()
                _LOGGER.debug("Found .model_name file: %s", original_name)
            except Exception:
                pass
        
        # Combine path and original name for detection
        search_target = (str(model_path) + " " + original_name).lower()
        if _find_file(model_path, ["*sense-voice*", "*sensevoice*"]) or \
           _find_file(model_path, ["model.onnx"]) and "sense" in search_target:
             model = _find_file(model_path, ["model.onnx", "*.onnx"]) # Usually just model.onnx
             if model:
                 return "sensevoice", {
                     "model": str(model),
                     "tokens": str(tokens) if tokens else None
                 }

        # 2. Transducer (Zipformer/Conformer/NeMo) - Streaming or Offline
        # Check FIRST because transducers also have encoder+decoder like Whisper, but additionally have joiner
        joiner = _find_file(model_path, ["*joiner.onnx", "*joiner*.onnx"])
        if joiner:
            encoder = _find_file(model_path, ["*encoder.onnx", "*encoder*.onnx"])
            decoder = _find_file(model_path, ["*decoder.onnx", "*decoder*.onnx"])
            if encoder and decoder:
                # Check for NeMo specific handling - same as Transducer but detected for logging
                if "nemo" in search_target or "nemo" in encoder.name.lower():
                     # NeMo might use tokenizer.model or vocab.txt instead of tokens.txt
                     nemo_tokens = tokens
                     if not nemo_tokens:
                        nemo_tokens = _find_file(model_path, ["tokenizer.model", "vocab.txt"])
                        
                     return "nemo-transducer", {
                        "encoder": str(encoder),
                        "decoder": str(decoder),
                        "joiner": str(joiner),
                        "tokens": str(nemo_tokens) if nemo_tokens else None
                    }
                
                return "transducer", {
                    "encoder": str(encoder),
                    "decoder": str(decoder),
                    "joiner": str(joiner),
                    "tokens": str(tokens) if tokens else None
                }
            
        # 3. Whisper (multilingual) - encoder + decoder but NO joiner
        encoder = _find_file(model_path, ["*encoder.onnx", "*encoder*.onnx"])
        decoder = _find_file(model_path, ["*decoder.onnx", "*decoder*.onnx"])
        if encoder and decoder:
            return "whisper", {
                "encoder": str(encoder),
                "decoder": str(decoder),
                "tokens": str(tokens) if tokens else None
            }
            
        # 4. Paraformer (Chinese)
        paraformer = _find_file(model_path, ["*paraformer*.onnx", "model.onnx"])
        if paraformer:
             return "paraformer", {
                 "model": str(paraformer),
                 "tokens": str(tokens) if tokens else None
             }

        # 5. Moonshine
        preprocessor = _find_file(model_path, ["preprocess.onnx", "*preprocess*.onnx"])
        if preprocessor:
             encoder = _find_file(model_path, ["*encoder.onnx"])
             uncached = _find_file(model_path, ["*uncached_decoder.onnx"])
             cached = _find_file(model_path, ["*cached_decoder.onnx"])
             if encoder and uncached and cached:
                 return "moonshine", {
                     "preprocessor": str(preprocessor),
                     "encoder": str(encoder),
                     "uncached_decoder": str(uncached),
                     "cached_decoder": str(cached),
                     "tokens": str(tokens) if tokens else None
                 }

        # 6. NeMo CTC (single model.onnx but with 'nemo' in name/path)
        if "nemo" in search_target:
            model = _find_file(model_path, ["model.onnx", "*.onnx"])
            if model:
                return "nemo-ctc", {
                    "model": str(model),
                    "tokens": str(tokens) if tokens else None
                }

        # 7. FireRedAsr (specific name pattern usually)
        if "fire-red" in search_target or "firered" in search_target:
             model = _find_file(model_path, ["model.onnx", "*.onnx"])
             if model:
                 return "fire-red-asr", {
                     "model": str(model),
                     "tokens": str(tokens) if tokens else None
                 }

        # 8. Wenet CTC
        if "wenet" in search_target:
             model = _find_file(model_path, ["final.onnx", "model.onnx", "*.onnx"])
             if model:
                 return "wenet-ctc", {
                     "model": str(model),
                     "tokens": str(tokens) if tokens else None
                 }

        # Default fallback (Generic CTC / Zipformer CTC)
        model = _find_file(model_path, ["*.onnx"])
        return "ctc", {
             "model": str(model) if model else None,
             "tokens": str(tokens) if tokens else None
        } 

    async def load(self) -> None:
        """Load the ASR model."""
        import sherpa_onnx

        _LOGGER.info("Loading ASR model from %s", self.config.model_path)

        model_type, config = self._detect_model_type(self.config.model_path)
        provider = self.config.provider if self.config.use_gpu else "cpu"
        
        # Define which model types support streaming (online) mode
        ONLINE_CAPABLE_TYPES = {"transducer", "paraformer", "online-transducer", "online-paraformer"}
        OFFLINE_ONLY_TYPES = {"whisper", "sensevoice", "moonshine", "ctc"}
        
        # Determine mode (Offline or Online)
        is_streaming = False
        
        if self.config.mode == "online":
            # Check if model supports online mode
            if model_type in OFFLINE_ONLY_TYPES:
                _LOGGER.warning(
                    "Model type '%s' does not support ONLINE (streaming) mode. "
                    "Falling back to OFFLINE mode. Remove ASR_MODE or set ASR_MODE=offline.",
                    model_type
                )
                is_streaming = False
            else:
                is_streaming = True
                _LOGGER.info("Forced ONLINE (streaming) mode via config")
        elif self.config.mode == "offline":
            is_streaming = False
            _LOGGER.info("Forced OFFLINE mode via config")
        else:
            # Auto-detect
            if model_type.startswith("online-"):
                is_streaming = True
            elif model_type in ONLINE_CAPABLE_TYPES and "streaming" in str(self.config.model_path).lower():
                is_streaming = True
            
            _LOGGER.info("Auto-detected mode: %s", "ONLINE" if is_streaming else "OFFLINE")

        _LOGGER.info("Detected %s model", model_type)
        self._model_type = model_type
        
        # --- Online (Streaming) Models ---
        if is_streaming:
            # Force model type to online variant if generic 'transducer' or 'paraformer' was returned
            if model_type == "transducer":
                 model_type = "online-transducer"
            elif model_type == "paraformer":
                 model_type = "online-paraformer"
                 
            self._model_type = model_type # Update internal type
            
            if model_type == "online-transducer":
                 # Zipformer uses transducer API
                self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=config["encoder"],
                    decoder=config["decoder"],
                    joiner=config["joiner"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "online-paraformer":
                self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                    model=config["model"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            else:
                 # This shouldn't happen if validation above is correct
                 raise ValueError(
                     f"Model type '{model_type}' does not support ONLINE mode. "
                     f"Set ASR_MODE=offline or use a streaming-capable model."
                 )

        # --- Offline Models ---
        else:
            if model_type == "sensevoice":
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=config["model"],
                    tokens=config["tokens"],
                    use_itn=True,
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "whisper":
                whisper_lang = self.config.language or ""
                _LOGGER.info("Whisper language: %s", whisper_lang if whisper_lang else "auto-detect")
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=config["encoder"],
                    decoder=config["decoder"],
                    tokens=config["tokens"],
                    language=whisper_lang,
                    task="transcribe",
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "moonshine":
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
                    preprocessor=config["preprocessor"],
                    encoder=config["encoder"],
                    uncached_decoder=config["uncached_decoder"],
                    cached_decoder=config["cached_decoder"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "transducer":
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=config["encoder"],
                    decoder=config["decoder"],
                    joiner=config["joiner"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "nemo-transducer":
                 # Use standard transducer loader but with (potentially) NeMo-specific tokens found above
                 self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=config["encoder"],
                    decoder=config["decoder"],
                    joiner=config["joiner"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "paraformer":
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    model=config["model"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "nemo-ctc":
                 self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                    model=config["model"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "fire-red-asr":
                 self._recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
                    model=config["model"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "wenet-ctc":
                 self._recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
                    model=config["model"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            elif model_type == "ctc":
                 # Fallback generic CTC (Zipformer CTC usually)
                 self._recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
                    model=config["model"],
                    tokens=config["tokens"],
                    num_threads=self.config.num_threads,
                    provider=provider,
                )
            else:
                raise ValueError(f"Unknown offline model type: {model_type}")

        self._model_name = _detect_model_name(self.config.model_path)
        _LOGGER.info("ASR model loaded: %s (%s - %s), provider: %s", 
                     self._model_name, 
                     model_type, 
                     "ONLINE" if is_streaming else "OFFLINE",
                     self.config.provider)

    def recognize(
        self, audio: np.ndarray, sample_rate: int, language: Optional[str] = None
    ) -> str:
        """Recognize speech from audio samples."""
        if self._recognizer is None:
            raise RuntimeError("ASR engine not loaded")

        # Resample logic (same as before)
        if sample_rate != ASR_SAMPLE_RATE:
            duration = len(audio) / sample_rate
            new_length = int(duration * ASR_SAMPLE_RATE)
            old_indices = np.linspace(0, len(audio) - 1, len(audio))
            new_indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(new_indices, old_indices, audio).astype(np.float32)
            sample_rate = ASR_SAMPLE_RATE

        if audio.dtype != np.float32:
             if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
             else:
                audio = audio.astype(np.float32)

        try:
            # Route to appropriate handler
            if str(self._model_type).startswith("online-"):
                return self._recognize_online(audio, sample_rate)
            else:
                return self._recognize_offline(audio, sample_rate)
        except RuntimeError as e:
            msg = str(e)
            if "invalid dimensions" in msg or "Mismatch" in msg:
                _LOGGER.error(
                    "ASR Runtime Error: %s. \n"
                    "This usually means a Streaming (Online) model is being run as Offline, or vice-versa.\n"
                    "Try setting ASR_MODE=online (or offline) in docker-compose.yml to match your model type.",
                    msg
                )
            raise

    def _recognize_offline(self, audio: np.ndarray, sample_rate: int) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return text

    def _recognize_online(self, audio: np.ndarray, sample_rate: int) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        text = self._recognizer.get_result(stream).text.strip()
        return text


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


def _find_file(model_path: Path, patterns: list[str]) -> Optional[Path]:
    """Find first file matching any of the patterns."""
    for pattern in patterns:
        matches = list(model_path.glob(pattern))
        if matches:
            return matches[0]
    return None


def _detect_model_name(model_path: Path) -> str:
    """
    Detect model name from ONNX filenames in the model directory.
    
    Examples:
    - large-v3-encoder.onnx -> whisper-large-v3
    - model.onnx in sense-voice dir -> sensevoice
    - vits-piper-de_DE-thorsten.onnx -> vits-piper-de_DE-thorsten
    """
    # Try to find encoder files (Whisper, Moonshine, etc.)
    for encoder in model_path.glob("*-encoder.onnx"):
        base = encoder.stem.replace("-encoder", "").replace(".int8", "")
        if base:
            return f"whisper-{base}" if not base.startswith("whisper") else base
    
    for encoder in model_path.glob("*encoder*.onnx"):
        base = encoder.stem.replace("encoder", "").replace("_", "").replace(".int8", "")
        if base and base not in ("", "-", "_"):
            return base
    
    # Try to find model.onnx and derive name from directory parent or nearby files
    model_onnx = model_path / "model.onnx"
    if not model_onnx.exists():
        model_onnx = model_path / "model.int8.onnx"
    
    if model_onnx.exists():
        # Look for a descriptive tokens file or check parent directory name
        for tokens in model_path.glob("*tokens*.txt"):
            base = tokens.stem.replace("-tokens", "").replace("_tokens", "")
            if base and base not in ("tokens",):
                return base
        
        # Check if parent directory has a more descriptive name
        parent = model_path.parent.name
        if parent not in ("models", "asr", "tts", "app"):
            return parent
    
    # Try to find any ONNX file with a descriptive name
    for onnx_file in model_path.glob("*.onnx"):
        name = onnx_file.stem.replace(".int8", "")
        # Skip generic names
        if name not in ("model", "encoder", "decoder", "joiner", "preprocessor", 
                        "cached_decoder", "uncached_decoder"):
            return name
    
    # Fallback to directory name but try parent if current is generic
    dir_name = model_path.name
    if dir_name in ("asr", "tts", "models"):
        parent_name = model_path.parent.name
        if parent_name not in ("app", "models", ""):
            return parent_name
    
    return dir_name


def _find_tokens(model_path: Path, base_name: Optional[str] = None) -> Optional[Path]:
    """Find tokens file with various naming conventions."""
    patterns = ["tokens.txt"]
    if base_name:
        # Handle naming like large-v3-tokens.txt
        patterns.insert(0, f"{base_name}-tokens.txt")
        patterns.insert(0, f"{base_name.replace('-encoder', '')}-tokens.txt")
    
    for pattern in patterns:
        tokens_file = model_path / pattern
        if tokens_file.exists():
            return tokens_file
    
    # Try glob patterns
    matches = list(model_path.glob("*-tokens.txt"))
    if matches:
        return matches[0]
    
    return None


class SherpaTTSEngine:
    """Sherpa-onnx TTS engine wrapper supporting multiple model types."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self._tts = None
        self.sample_rate = TTS_SAMPLE_RATE
        self._model_type: Optional[str] = None
        self._model_name: Optional[str] = None

    @property
    def model_type(self) -> Optional[str]:
        """Return detected model type (vits, matcha, kokoro)."""
        return self._model_type

    @property
    def model_name(self) -> str:
        """Return cleaned model name (without vits-/sherpa-onnx- prefix)."""
        name = self._model_name or self.config.model_path.name
        # Strip common prefixes
        for prefix in ["sherpa-onnx-", "sherpa_onnx_", "vits-"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name

    def _detect_model_type(self, model_path: Path) -> Tuple[str, dict]:
        """
        Detect TTS model type and return (type_name, config_dict).
        
        Supports:
        - VITS/Piper: model.onnx + tokens.txt (+ optional espeak-ng-data, lexicon.txt, dict/)
        - Matcha: model.onnx + tokens.txt (+ optional vocoder.onnx)
        - Kokoro: model.onnx + tokens.txt + voices.bin (multi-speaker)
        """
        
        # Find model file
        model_onnx = _find_file(model_path, ["model.onnx", "model.int8.onnx"])
        if not model_onnx:
            # Try any .onnx file
            onnx_files = list(model_path.glob("*.onnx"))
            # Exclude vocoder files
            onnx_files = [f for f in onnx_files if "vocoder" not in f.name.lower()]
            if onnx_files:
                model_onnx = onnx_files[0]
        
        if not model_onnx:
            raise ValueError(f"No TTS ONNX model found in {model_path}")
        
        tokens = _find_tokens(model_path)
        if not tokens:
            raise ValueError(f"tokens.txt not found in {model_path}")
        
        path_lower = str(model_path).lower()
        
        # Detect Kokoro (has voices.bin)
        voices_bin = model_path / "voices.bin"
        if voices_bin.exists() or "kokoro" in path_lower:
            return "kokoro", {
                "model": str(model_onnx),
                "tokens": str(tokens),
                "voices": str(voices_bin) if voices_bin.exists() else "",
            }
        
        # Detect Matcha (check for separate vocoder or matcha in path)
        vocoder = _find_file(model_path, ["vocoder.onnx", "hifigan.onnx", "*vocoder*.onnx"])
        if vocoder or "matcha" in path_lower:
            return "matcha", {
                "model": str(model_onnx),
                "tokens": str(tokens),
                "vocoder": str(vocoder) if vocoder else "",
            }
        
        # Default: VITS/Piper (most common)
        data_dir = model_path / "espeak-ng-data"
        lexicon = model_path / "lexicon.txt"
        dict_dir = model_path / "dict"
        
        return "vits", {
            "model": str(model_onnx),
            "tokens": str(tokens),
            "data_dir": str(data_dir) if data_dir.exists() else "",
            "lexicon": str(lexicon) if lexicon.exists() else "",
            "dict_dir": str(dict_dir) if dict_dir.exists() else "",
        }

    async def load(self) -> None:
        """Load the TTS model."""
        import sherpa_onnx

        _LOGGER.info("Loading TTS model from %s", self.config.model_path)

        model_type, config = self._detect_model_type(self.config.model_path)
        provider = self.config.provider if self.config.use_gpu else "cpu"
        
        _LOGGER.info("Detected %s TTS model", model_type)
        
        if model_type == "vits":
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=config["model"],
                        tokens=config["tokens"],
                        data_dir=config["data_dir"],
                        lexicon=config["lexicon"],
                        dict_dir=config["dict_dir"],
                    ),
                    provider=provider,
                    num_threads=self.config.num_threads,
                ),
                max_num_sentences=1,
            )
            self._tts = sherpa_onnx.OfflineTts(tts_config)
        
        elif model_type == "matcha":
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                        acoustic_model=config["model"],
                        vocoder=config["vocoder"],
                        tokens=config["tokens"],
                    ),
                    provider=provider,
                    num_threads=self.config.num_threads,
                ),
                max_num_sentences=1,
            )
            self._tts = sherpa_onnx.OfflineTts(tts_config)
        
        elif model_type == "kokoro":
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=config["model"],
                        tokens=config["tokens"],
                        voices=config["voices"],
                    ),
                    provider=provider,
                    num_threads=self.config.num_threads,
                ),
                max_num_sentences=1,
            )
            self._tts = sherpa_onnx.OfflineTts(tts_config)
        
        else:
            raise ValueError(f"Unknown TTS model type: {model_type}")
        
        self._model_type = model_type
        self._model_name = _detect_model_name(self.config.model_path)
        self.sample_rate = self._tts.sample_rate
        _LOGGER.info("TTS loaded: %s (%s), sample rate: %d", self._model_name, model_type, self.sample_rate)

    def _smart_split(self, text: str, max_len: int = 200) -> list[str]:
        """Recursively split text into chunks smaller than max_len."""
        # Pre-clean: Remove markdown formatting that might confuse TTS
        # Remove bold/italic markers
        text = text.replace("**", "").replace("*", "")
        
        if len(text) <= max_len:
            return [text]
        
        import re
        
        # 1. Split by newlines (important for lists)
        # Keep newlines as separators to preserve structure if needed, or just split
        parts = re.split(r'(\n+)', text)
        chunks = []
        current = ""
        
        for part in parts:
            if len(current) + len(part) <= max_len:
                current += part
            else:
                if current.strip():
                    chunks.append(current)
                current = part
        if current.strip():
            chunks.append(current)
            
        # 2. Split by sentence endings (.?!)
        # Recurse on chunks that are still too big
        better_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_len:
                better_chunks.append(chunk)
                continue
                
            parts = re.split(r'([.?!;]+\s+)', chunk)
            current = ""
            for part in parts:
                if len(current) + len(part) <= max_len:
                    current += part
                else:
                    if current.strip():
                        better_chunks.append(current)
                    current = part
            if current.strip():
                better_chunks.append(current)

        # 3. Split by clauses (,:—) for remaining big chunks
        final_chunks = []
        for chunk in better_chunks:
            if len(chunk) <= max_len:
                final_chunks.append(chunk)
                continue
                
            # Split by comma/colon/dash
            sub_parts = re.split(r'([,:—]+\s+)', chunk)
            sub_current = ""
            for part in sub_parts:
                if len(sub_current) + len(part) <= max_len:
                    sub_current += part
                else:
                    if sub_current.strip():
                         # If even sub-chunk is too big, split by space
                        if len(sub_current) > max_len:
                            words = sub_current.split(' ')
                            w_current = ""
                            for word in words:
                                if len(w_current) + len(word) + 1 <= max_len:
                                    w_current += (word + " ")
                                else:
                                    if w_current.strip():
                                        final_chunks.append(w_current.strip())
                                    w_current = word + " "
                            if w_current.strip():
                                final_chunks.append(w_current.strip())
                        else:
                            final_chunks.append(sub_current)
                    sub_current = part
            if sub_current.strip():
                 # Check last piece
                if len(sub_current) > max_len:
                    words = sub_current.split(' ')
                    w_current = ""
                    for word in words:
                        if len(w_current) + len(word) + 1 <= max_len:
                            w_current += (word + " ")
                        else:
                            if w_current.strip():
                                final_chunks.append(w_current.strip())
                            w_current = word + " "
                    if w_current.strip():
                        final_chunks.append(w_current.strip())
                else:
                    final_chunks.append(sub_current)
                    
        return [c for c in final_chunks if c.strip()]

    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> np.ndarray:
        """Synthesize speech from text."""
        if self._tts is None:
            raise RuntimeError("TTS engine not loaded")

        sid = speaker_id if speaker_id is not None else self.config.speaker_id

        # Smart split text into chunks < 200 chars to avoid model crashes
        # Lowered limit to 200 for extra safety with markdown/lists
        sentences = self._smart_split(text, max_len=200)
        
        # If somehow we still have no sentences
        if not sentences and text.strip():
            sentences = [text]

        # Synthesize each chunk
        all_samples = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            try:
                audio = self._tts.generate(
                    sentence,
                    sid=sid,
                    speed=self.config.speed,
                )
                if len(audio.samples) > 0:
                    all_samples.append(audio.samples)
                    
                    # Add a small silence between chunks
                    silence_duration = 0.1
                    silence_samples = int(self.sample_rate * silence_duration)
                    all_samples.append(np.zeros(silence_samples, dtype=np.float32))
            except Exception as e:
                # Log but continue with other chunks if one fails
                _LOGGER.error("Failed to synthesize chunk '%s': %s", sentence[:50], e)

        if not all_samples:
             return np.array([], dtype=np.float32)

        # Concatenate all parts
        return np.concatenate(all_samples)

