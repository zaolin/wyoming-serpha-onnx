# Wyoming Sherpa-ONNX

Wyoming protocol server providing **ASR (Speech-to-Text)** and **TTS (Text-to-Speech)** using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on NVIDIA Jetson Orin NX with GPU acceleration.

## Features

- 🎤 **ASR**: Supports all sherpa-onnx model types (Whisper, SenseVoice, Moonshine, Transducer, Paraformer, CTC)
- 🔊 **TTS**: Supports VITS/Piper, Matcha, and Kokoro models
- 🚀 **GPU Accelerated**: CUDA 12.6 + cuDNN 9 on Jetson Orin NX
- 🏠 **Home Assistant Ready**: Wyoming protocol compatible
- 🔄 **Model Agnostic**: Auto-detects model type from file structure

## Quick Start

### 1. Download Models

```bash
# Make the script executable
chmod +x download_models.sh

# List available models
./download_models.sh --list

# Download default models (Whisper large-v3 + German Piper TTS)
./download_models.sh all

# Or choose specific models
./download_models.sh asr sensevoice        # 99 languages
./download_models.sh tts kokoro-en         # High quality English
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

## Supported Models

### ASR Models

| Model | Type | Languages | Size | Use Case |
|-------|------|-----------|------|----------|
| `whisper-large-v3` | Whisper | 99 | 3.1GB | Best accuracy |
| `whisper-large-v3-turbo` | Whisper | 99 | 1.6GB | Fast + accurate |
| `whisper-medium` | Whisper | 99 | 1.5GB | Balanced |
| `whisper-small` | Whisper | 99 | 483MB | Fast |
| `sensevoice` | SenseVoice | 99 | 450MB | Multilingual |
| `moonshine-base` | Moonshine | English | 200MB | Ultra fast |
| `paraformer-zh` | Paraformer | Chinese | 220MB | Chinese focus |
| `zipformer-en` | Transducer | English | 65MB | Streaming |

### TTS Models

| Model | Type | Language | Quality |
|-------|------|----------|---------|
| `piper-de-thorsten` | VITS/Piper | German | High |
| `piper-en-lessac` | VITS/Piper | English | Medium |
| `piper-fr-siwis` | VITS/Piper | French | Medium |
| `kokoro-en` | Kokoro | English | Very High |
| `kokoro-multi` | Kokoro | Multi | Very High |
| `melo-zh-en` | VITS | Chinese/English | High |
| `matcha-en-ljspeech` | Matcha | English | High |

See all models: `./download_models.sh --list`

## Configuration

### docker-compose.yml

```yaml
services:
  sherpa-onnx:
    environment:
      # Model selection (use model directory names from ./models/)
      ASR_MODEL: asr          # ASR model subdirectory
      TTS_MODEL: tts          # TTS model subdirectory
      TTS_SPEAKER_ID: 0       # Speaker ID for multi-speaker models
      
      # Server ports
      ASR_PORT: 10300
      TTS_PORT: 10400
      
      # Performance
      USE_GPU: "true"
      NUM_THREADS: 4
```

### Custom Models

You can use any model from [sherpa-onnx pretrained models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html):

1. Download and extract to `./models/asr/` or `./models/tts/`
2. The engine auto-detects the model type based on file structure
3. Restart the container

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Wyoming Server                        │
│  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │   ASR Handler       │  │      TTS Handler        │   │
│  │   Port 10300        │  │      Port 10400         │   │
│  └──────────┬──────────┘  └───────────┬─────────────┘   │
│             │                         │                  │
│  ┌──────────▼─────────────────────────▼─────────────┐   │
│  │              Sherpa-ONNX Engine                   │   │
│  │  ┌─────────────────┐  ┌────────────────────┐     │   │
│  │  │ ASR Recognizer  │  │  TTS Synthesizer   │     │   │
│  │  │ (Auto-detect)   │  │  (Auto-detect)     │     │   │
│  │  └─────────────────┘  └────────────────────┘     │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │              CUDA / GPU Acceleration              │   │
│  │              (Jetson Orin NX)                     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
wyoming-serpha-onnx/
├── wyoming_sherpa_onnx/
│   ├── __init__.py          # Package init
│   ├── __main__.py          # Server entry point
│   ├── engine.py            # Model-agnostic ASR/TTS engines
│   ├── asr_handler.py       # Wyoming ASR protocol handler
│   └── tts_handler.py       # Wyoming TTS protocol handler
├── models/
│   ├── asr/                 # ASR model files
│   └── tts/                 # TTS model files
├── Dockerfile               # Jetson-optimized build
├── docker-compose.yml       # Service configuration
├── download_models.sh       # Model downloader
├── pyproject.toml           # Python project config
└── README.md
```

## Troubleshooting

### Check logs
```bash
docker compose logs -f sherpa-onnx
```

### Verify GPU access
```bash
docker compose exec sherpa-onnx nvidia-smi
```

### Test ASR
```bash
# Using netcat
echo '{"type":"transcribe"}' | nc localhost 10300
```

### Model detection issues
The engine auto-detects model types based on files:
- **Whisper**: `*-encoder.onnx` + `*-decoder.onnx`
- **SenseVoice/Paraformer**: `model.onnx` + `tokens.txt`
- **Moonshine**: `preprocessor.onnx` + `encoder.onnx` + `*_decoder.onnx`
- **Transducer**: `encoder.onnx` + `decoder.onnx` + `joiner.onnx`
- **VITS/Piper**: `model.onnx` + `tokens.txt` + `espeak-ng-data/`
- **Kokoro**: `model.onnx` + `voices.bin`
- **Matcha**: `model.onnx` + `vocoder.onnx`

## License

MIT License - see [LICENSE](LICENSE)

## Credits

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - ONNX speech recognition/synthesis
- [wyoming](https://github.com/home-assistant/wyoming) - Voice assistant protocol
- [Piper](https://github.com/rhasspy/piper) - Neural text-to-speech
