#!/bin/bash
# Download models for Wyoming sherpa-onnx
# Usage: ./download_models.sh [asr|tts|all] [model_name]
#
# Examples:
#   ./download_models.sh all                    # Download default ASR + TTS
#   ./download_models.sh asr whisper-large-v3   # Download Whisper large-v3
#   ./download_models.sh tts kokoro-en          # Download Kokoro English TTS

set -e

MODELS_DIR="${MODELS_DIR:-./models}"
GITHUB_RELEASES="https://github.com/k2-fsa/sherpa-onnx/releases/download"
HF_BASE="https://huggingface.co/k2-fsa/sherpa-onnx-tts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ============================================================================
# ASR Models
# ============================================================================

declare -A ASR_MODELS=(
    # Whisper models
    ["whisper-tiny"]="asr-models/sherpa-onnx-whisper-tiny"
    ["whisper-base"]="asr-models/sherpa-onnx-whisper-base"
    ["whisper-small"]="asr-models/sherpa-onnx-whisper-small"
    ["whisper-medium"]="asr-models/sherpa-onnx-whisper-medium"
    ["whisper-large-v3"]="asr-models/sherpa-onnx-whisper-large-v3"
    ["whisper-large-v3-turbo"]="asr-models/sherpa-onnx-whisper-large-v3-turbo"
    
    # SenseVoice (multilingual)
    ["sensevoice"]="asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    
    # Moonshine (fast, lightweight)
    ["moonshine-tiny"]="asr-models/sherpa-onnx-moonshine-tiny-en-int8"
    ["moonshine-base"]="asr-models/sherpa-onnx-moonshine-base-en-int8"
    
    # Paraformer (Chinese focused)
    ["paraformer-zh"]="asr-models/sherpa-onnx-paraformer-zh-2023-09-14"
    
    # Zipformer Transducer (streaming capable)
    ["zipformer-en"]="asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
    ["zipformer-bilingual"]="asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
)

# ============================================================================
# TTS Models
# ============================================================================

declare -A TTS_MODELS=(
    # Piper VITS models (various languages)
    ["piper-de-thorsten"]="tts-models/vits-piper-de_DE-thorsten-high"
    ["piper-en-lessac"]="tts-models/vits-piper-en_US-lessac-medium"
    ["piper-en-libritts"]="tts-models/vits-piper-en_US-libritts_r-medium"
    ["piper-fr-siwis"]="tts-models/vits-piper-fr_FR-siwis-medium"
    ["piper-es-davefx"]="tts-models/vits-piper-es_ES-davefx-medium"
    ["piper-it-riccardo"]="tts-models/vits-piper-it_IT-riccardo-x_low"
    ["piper-nl-mls"]="tts-models/vits-piper-nl_NL-mls-medium"
    ["piper-pl-gosia"]="tts-models/vits-piper-pl_PL-gosia-medium"
    ["piper-pt-faber"]="tts-models/vits-piper-pt_BR-faber-medium"
    ["piper-ru-irina"]="tts-models/vits-piper-ru_RU-irina-medium"
    ["piper-zh-huayan"]="tts-models/vits-piper-zh_CN-huayan-medium"
    
    # Kokoro models (high quality, multi-speaker)
    ["kokoro-en"]="tts-models/kokoro-en-v0_19"
    ["kokoro-multi"]="tts-models/kokoro-multi-lang-v1_0"
    
    # VITS MeloTTS
    ["melo-zh-en"]="tts-models/vits-melo-tts-zh_en"
    
    # Matcha models
    ["matcha-en-ljspeech"]="tts-models/matcha-icefall-en_US-ljspeech"
)

# Default models
DEFAULT_ASR="whisper-large-v3"
DEFAULT_TTS="piper-de-thorsten"

list_models() {
    echo ""
    echo "Available ASR Models:"
    echo "====================="
    for model in "${!ASR_MODELS[@]}"; do
        if [[ "$model" == "$DEFAULT_ASR" ]]; then
            echo "  $model (default)"
        else
            echo "  $model"
        fi
    done | sort
    
    echo ""
    echo "Available TTS Models:"
    echo "====================="
    for model in "${!TTS_MODELS[@]}"; do
        if [[ "$model" == "$DEFAULT_TTS" ]]; then
            echo "  $model (default)"
        else
            echo "  $model"
        fi
    done | sort
    echo ""
}

download_model() {
    local model_type=$1
    local model_name=$2
    local target_dir=$3
    local model_path
    
    if [[ "$model_type" == "asr" ]]; then
        model_path="${ASR_MODELS[$model_name]}"
        [[ -z "$model_path" ]] && error "Unknown ASR model: $model_name. Use --list to see available models."
    else
        model_path="${TTS_MODELS[$model_name]}"
        [[ -z "$model_path" ]] && error "Unknown TTS model: $model_name. Use --list to see available models."
    fi
    
    local url="${GITHUB_RELEASES}/${model_path}.tar.bz2"
    local tarball_name=$(basename "$model_path").tar.bz2
    local dir_name=$(basename "$model_path")
    
    info "Downloading $model_name from $url"
    
    mkdir -p "$target_dir"
    cd "$target_dir"
    
    # Download if not exists
    if [[ ! -f "$tarball_name" ]]; then
        if ! curl -L -o "$tarball_name" "$url" 2>/dev/null; then
            # Try alternative URL format
            url="${GITHUB_RELEASES}/v1.10.30/${dir_name}.tar.bz2"
            info "Trying alternative URL: $url"
            curl -L -o "$tarball_name" "$url" || error "Failed to download $model_name"
        fi
    else
        warn "Tarball already exists, skipping download"
    fi
    
    # Extract
    info "Extracting $tarball_name"
    tar -xjf "$tarball_name"
    
    # Move contents to target (asr or tts subdirectory)
    if [[ "$model_type" == "asr" ]]; then
        rm -rf asr
        mv "$dir_name" asr
    else
        rm -rf tts
        mv "$dir_name" tts
    fi
    
    # Cleanup tarball
    rm -f "$tarball_name"
    
    success "Downloaded $model_name to $target_dir/$model_type"
    cd - > /dev/null
}

download_asr() {
    local model="${1:-$DEFAULT_ASR}"
    info "Downloading ASR model: $model"
    download_model "asr" "$model" "$MODELS_DIR"
}

download_tts() {
    local model="${1:-$DEFAULT_TTS}"
    info "Downloading TTS model: $model"
    download_model "tts" "$model" "$MODELS_DIR"
}

show_help() {
    echo "Wyoming Sherpa-ONNX Model Downloader"
    echo ""
    echo "Usage: $0 [command] [model_name]"
    echo ""
    echo "Commands:"
    echo "  all [asr_model] [tts_model]  Download ASR and TTS models (default: $DEFAULT_ASR, $DEFAULT_TTS)"
    echo "  asr [model_name]             Download ASR model (default: $DEFAULT_ASR)"
    echo "  tts [model_name]             Download TTS model (default: $DEFAULT_TTS)"
    echo "  --list                       List all available models"
    echo "  --help                       Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  MODELS_DIR                   Directory to store models (default: ./models)"
    echo ""
    echo "Examples:"
    echo "  $0 all                           # Download default models"
    echo "  $0 asr whisper-large-v3          # Download Whisper large-v3"
    echo "  $0 asr sensevoice                # Download SenseVoice (99 languages)"
    echo "  $0 tts kokoro-en                 # Download Kokoro English TTS"
    echo "  $0 tts piper-de-thorsten         # Download German Piper TTS"
    echo ""
}

# Main
case "${1:-all}" in
    all)
        download_asr "${2:-$DEFAULT_ASR}"
        download_tts "${3:-$DEFAULT_TTS}"
        echo ""
        success "All models downloaded to $MODELS_DIR"
        echo "  ASR: $MODELS_DIR/asr"
        echo "  TTS: $MODELS_DIR/tts"
        ;;
    asr)
        download_asr "${2:-$DEFAULT_ASR}"
        ;;
    tts)
        download_tts "${2:-$DEFAULT_TTS}"
        ;;
    --list|-l)
        list_models
        ;;
    --help|-h)
        show_help
        ;;
    *)
        error "Unknown command: $1. Use --help for usage."
        ;;
esac
