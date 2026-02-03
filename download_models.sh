#!/bin/bash
# Download models for Wyoming sherpa-onnx
# Usage: ./download_models.sh [asr|tts|all]

set -e

MODELS_DIR="${MODELS_DIR:-./models}"
GITHUB_RELEASES="https://github.com/k2-fsa/sherpa-onnx/releases/download"

# Default models
ASR_MODEL="${ASR_MODEL:-sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17}"
TTS_MODEL="${TTS_MODEL:-vits-piper-de_DE-thorsten-high}"

download_asr() {
    echo "Downloading ASR model: $ASR_MODEL"
    mkdir -p "$MODELS_DIR/asr"
    cd "$MODELS_DIR/asr"
    
    if [[ ! -f "model.onnx" ]]; then
        wget -q --show-progress "${GITHUB_RELEASES}/asr-models/${ASR_MODEL}.tar.bz2"
        tar xf "${ASR_MODEL}.tar.bz2"
        mv "${ASR_MODEL}"/* . 2>/dev/null || true
        rm -rf "${ASR_MODEL}" "${ASR_MODEL}.tar.bz2"
        echo "ASR model downloaded successfully"
    else
        echo "ASR model already exists"
    fi
    cd - > /dev/null
}

download_tts() {
    echo "Downloading TTS model: $TTS_MODEL"
    mkdir -p "$MODELS_DIR/tts"
    cd "$MODELS_DIR/tts"
    
    if [[ ! -f "*.onnx" ]] && [[ ! -f "model.onnx" ]]; then
        wget -q --show-progress "${GITHUB_RELEASES}/tts-models/${TTS_MODEL}.tar.bz2"
        tar xf "${TTS_MODEL}.tar.bz2"
        mv "${TTS_MODEL}"/* . 2>/dev/null || true
        rm -rf "${TTS_MODEL}" "${TTS_MODEL}.tar.bz2"
        echo "TTS model downloaded successfully"
    else
        echo "TTS model already exists"
    fi
    cd - > /dev/null
}

case "${1:-all}" in
    asr)
        download_asr
        ;;
    tts)
        download_tts
        ;;
    all)
        download_asr
        download_tts
        ;;
    *)
        echo "Usage: $0 [asr|tts|all]"
        exit 1
        ;;
esac

echo "Done!"
