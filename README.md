# Wyoming Sherpa-ONNX

Wyoming protocol server providing **ASR (Speech-to-Text)** and **TTS (Text-to-Speech)** using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on NVIDIA Jetson Orin NX with GPU acceleration.

## Features

- 🎤 **ASR**: Multilingual speech recognition (SenseVoice, Whisper, etc.)
- 🔊 **TTS**: High-quality text-to-speech (VITS, Piper voices)
- 🚀 **GPU Accelerated**: CUDA 12.6 + cuDNN 9 on Jetson Orin NX
- 🏠 **Home Assistant**: Compatible with Wyoming protocol integration

## Quick Start

### 1. Download Models

```bash
# Create model directories
mkdir -p models/asr models/tts

# Download SenseVoice ASR model (multilingual, 99 languages)
cd models/asr
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
mv sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/* .
rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17*

# Download German TTS model (Thorsten voice)
cd ../tts
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-thorsten-high.tar.bz2
tar xf vits-piper-de_DE-thorsten-high.tar.bz2
mv vits-piper-de_DE-thorsten-high/* .
rm -rf vits-piper-de_DE-thorsten-high*
```

### 2. Build and Run

```bash
docker compose build
docker compose up -d
```

### 3. Configure in Home Assistant

Add two Wyoming integrations:

1. **ASR (Speech-to-Text)**
   - Host: `<your-jetson-ip>`
   - Port: `10300`

2. **TTS (Text-to-Speech)**
   - Host: `<your-jetson-ip>`
   - Port: `10400`

## Configuration

All settings are configurable via environment variables in `docker-compose.yml`:

| Variable | Description | Default |
|----------|-------------|---------|
| `ASR_MODEL` | ASR model name | `sensevoice` |
| `TTS_MODEL` | TTS model name | `vits-piper-de_DE-thorsten-high` |
| `TTS_SPEAKER_ID` | Speaker ID for multi-speaker models | `0` |
| `USE_GPU` | Enable GPU acceleration | `true` |
| `ASR_URI` | ASR server URI | `tcp://0.0.0.0:10300` |
| `TTS_URI` | TTS server URI | `tcp://0.0.0.0:10400` |

## Available Models

### ASR Models

| Model | Languages | Description |
|-------|-----------|-------------|
| `sensevoice` | 99+ | Fast multilingual (recommended) |
| `whisper-tiny` | 99+ | OpenAI Whisper tiny |
| `whisper-base` | 99+ | OpenAI Whisper base |

### TTS Models (German)

| Model | Speaker | Description |
|-------|---------|-------------|
| `vits-piper-de_DE-thorsten-high` | Male | High-quality German |
| `vits-piper-de_DE-eva_k-x_low` | Female | German female voice |

Browse all models: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

## Requirements

- NVIDIA Jetson Orin NX
- JetPack 6.x (L4T 36.4.x)
- Docker with NVIDIA runtime

## License

MIT License - See [LICENSE](LICENSE) for details.
