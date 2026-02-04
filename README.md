# Wyoming Sherpa-ONNX

Wyoming protocol server providing **ASR (Speech-to-Text)** and **TTS (Text-to-Speech)** using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on NVIDIA Jetson Orin NX with GPU acceleration.

## Features

- 🎤 **ASR**: All sherpa-onnx models (Whisper, SenseVoice, Moonshine, Transducer, Paraformer, CTC)
- 🔊 **TTS**: VITS/Piper, Matcha, and Kokoro models
- 🚀 **GPU Accelerated**: CUDA 12.6 + cuDNN 9 on Jetson Orin NX
- 🏠 **Home Assistant Ready**: Wyoming protocol compatible
- 🔄 **Model Agnostic**: Auto-detects model type from file structure
- 🧠 **Memory Optimized**: Separate GPU/CPU control for ASR and TTS
- 🌍 **Multilingual**: Auto-detects supported languages from model names

## Quick Start

### 1. Download Models

```bash
chmod +x download_models.sh

# List all available models (384 ASR + 594 TTS)
./download_models.sh --list-asr
./download_models.sh --list-tts

# Search for specific models
./download_models.sh --list-asr whisper
./download_models.sh --list-tts piper-de

# Download specific models
./download_models.sh asr sherpa-onnx-whisper-large-v3
./download_models.sh tts vits-piper-de_DE-thorsten-high

# Or download defaults
./download_models.sh all
```

### 2. Build and Run

```bash
docker compose up -d --build
```

### 3. Configure Home Assistant

Add to `configuration.yaml`:

```yaml
wyoming:
  - name: "Sherpa ASR"
    host: <jetson-ip>
    port: 10300

  - name: "Sherpa TTS"
    host: <jetson-ip>
    port: 10400
```

## GPU and Memory Configuration

### Separate GPU Control

You can run ASR and TTS on different devices to optimize VRAM usage:

```yaml
# docker-compose.yml
environment:
  - USE_GPU=true        # Default for both
  - ASR_GPU=true        # ASR on GPU (large models benefit most)
  - TTS_GPU=false       # TTS on CPU (smaller models, saves VRAM)
```

**Example configurations:**

| Configuration | VRAM Usage | Best For |
|--------------|------------|----------|
| `ASR_GPU=true, TTS_GPU=true` | High | Fastest inference |
| `ASR_GPU=true, TTS_GPU=false` | Medium | Large ASR models (Whisper large-v3) |
| `ASR_GPU=false, TTS_GPU=true` | Low | Fast TTS response |
| `ASR_GPU=false, TTS_GPU=false` | None | CPU-only mode |

### Automatic Memory Management

The server automatically:
- Estimates model sizes before loading
- Checks available GPU memory
- Falls back to CPU if insufficient VRAM
- Logs memory usage decisions

## Supported Models

### ASR Models (384 available)

| Model | Languages | Size | Notes |
|-------|-----------|------|-------|
| `sherpa-onnx-whisper-large-v3` | 99 | 3.1GB | Best accuracy |
| `sherpa-onnx-whisper-turbo` | 99 | 1.6GB | Fast + accurate |
| `sherpa-onnx-whisper-small` | 99 | 483MB | Balanced |
| `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-*` | 5 | 450MB | Asian languages |
| `sherpa-onnx-moonshine-base-en-int8` | EN | 100MB | Ultra fast |
| `sherpa-onnx-paraformer-zh-*` | ZH | 220MB | Chinese |

### TTS Models (594 available)

| Model | Language | Quality |
|-------|----------|---------|
| `vits-piper-de_DE-thorsten-high` | German | High |
| `vits-piper-en_US-lessac-medium` | English | Medium |
| `vits-piper-*-int8` | Various | Quantized (saves memory) |
| `kokoro-multi-lang-v1_0` | Multi | Very High |
| `vits-coqui-*` | Various | Classic |

**Browse all models:**
- ASR: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
- TTS: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL_PATH` | `/app/models/asr` | Path to ASR model |
| `TTS_MODEL_PATH` | `/app/models/tts` | Path to TTS model |
| `TTS_SPEAKER_ID` | `0` | Speaker ID for multi-speaker models |
| `USE_GPU` | `true` | Enable GPU for both ASR and TTS |
| `ASR_GPU` | (use `USE_GPU`) | Override GPU setting for ASR only |
| `TTS_GPU` | (use `USE_GPU`) | Override GPU setting for TTS only |
| `ASR_URI` | `tcp://0.0.0.0:10300` | ASR server URI |
| `TTS_URI` | `tcp://0.0.0.0:10400` | TTS server URI |
| `ASR_ONLY` | `false` | Run only ASR server |
| `TTS_ONLY` | `false` | Run only TTS server |

### CLI Arguments

```bash
python -m wyoming_sherpa_onnx \
  --asr-model /path/to/asr \
  --tts-model /path/to/tts \
  --asr-gpu true \
  --tts-gpu false \
  --speaker-id 0 \
  --debug
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Wyoming Server                        │
│  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │   ASR Handler       │  │      TTS Handler        │   │
│  │   Port 10300        │  │      Port 10400         │   │
│  │   (in-memory audio) │  │   (streaming chunks)    │   │
│  └──────────┬──────────┘  └───────────┬─────────────┘   │
│             │                         │                  │
│  ┌──────────▼─────────────────────────▼─────────────┐   │
│  │              Sherpa-ONNX Engine                   │   │
│  │  ┌─────────────────┐  ┌────────────────────┐     │   │
│  │  │ ASR (GPU/CPU)   │  │  TTS (GPU/CPU)     │     │   │
│  │  │ Auto-detect     │  │  Auto-detect       │     │   │
│  │  └─────────────────┘  └────────────────────┘     │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │         ONNX Runtime (CUDA / CPU Provider)        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Check logs
```bash
docker compose logs -f sherpa-onnx
```

### Memory issues
If you see `MemoryError`, the server will automatically fall back to CPU. You can also:
- Use smaller models (e.g., `whisper-small` instead of `large-v3`)
- Use quantized models (e.g., `*-int8` variants)
- Set `TTS_GPU=false` to run TTS on CPU

### Verify GPU access
```bash
docker compose exec sherpa-onnx nvidia-smi
```

### Model detection
The engine auto-detects model types:
- **Whisper**: `*-encoder.onnx` + `*-decoder.onnx`
- **SenseVoice/Paraformer**: `model.onnx` + `tokens.txt`
- **Moonshine**: `preprocessor.onnx` + `encoder.onnx` + `*_decoder.onnx`
- **Transducer**: `encoder.onnx` + `decoder.onnx` + `joiner.onnx`
- **VITS/Piper**: `model.onnx` + `tokens.txt` (+ optional `espeak-ng-data/`)
- **Kokoro**: `model.onnx` + `voices.bin`
- **Matcha**: `model.onnx` + `vocoder.onnx`

## License

MIT License - see [LICENSE](LICENSE)

## Credits

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - ONNX speech recognition/synthesis
- [wyoming](https://github.com/home-assistant/wyoming) - Voice assistant protocol
- [Piper](https://github.com/rhasspy/piper) - Neural text-to-speech
