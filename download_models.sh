#!/bin/bash
# Download models for Wyoming sherpa-onnx
# 
# Models are downloaded from GitHub releases:
#   ASR: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
#   TTS: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
#
# IMPORTANT: Models must be compatible with the sherpa-onnx version installed!
# Check SHERPA_ONNX_VERSION below and match it to your Docker image.
#
# Usage:
#   ./download_models.sh --list-asr              # List available ASR models
#   ./download_models.sh --list-tts              # List available TTS models
#   ./download_models.sh asr <model-name>        # Download specific ASR model
#   ./download_models.sh tts <model-name>        # Download specific TTS model
#   ./download_models.sh all                     # Download defaults

set -e

# ============================================================================
#  VERSION CONFIGURATION - Must match the sherpa-onnx version in Dockerfile!
# ============================================================================
SHERPA_ONNX_VERSION="1.12.23"

# Models directory and GitHub base URLs
MODELS_DIR="${MODELS_DIR:-./models}"
GITHUB_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
GITHUB_API="https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ============================================================================
# Version-specific model recommendations
# Models are published to asr-models/tts-models tags, but not all work with
# all versions. This list shows verified compatible models per version.
# ============================================================================

# Models known to work with v1.12.x
MODELS_V1_12_ASR="sherpa-onnx-whisper-turbo sherpa-onnx-whisper-large-v3 sherpa-onnx-whisper-small sherpa-onnx-moonshine-base-en-int8"
MODELS_V1_12_TTS="vits-piper-de_DE-thorsten-high vits-piper-en_US-lessac-medium kokoro-multi-lang-v1_0"

# Models requiring v1.13+ (newer SenseVoice format)
MODELS_V1_13_ONLY="sherpa-onnx-sense-voice"

# Default models (stable on current version)
DEFAULT_ASR="sherpa-onnx-whisper-turbo"
DEFAULT_TTS="vits-piper-de_DE-thorsten-high"

# Check if model requires newer version
check_model_compatibility() {
    local model_name=$1
    
    # Check for known v1.13+ only models
    for pattern in $MODELS_V1_13_ONLY; do
        if [[ "$model_name" == *"$pattern"* ]]; then
            local major_minor="${SHERPA_ONNX_VERSION%.*}"
            if [[ "$major_minor" < "1.13" ]]; then
                warn "Model '$model_name' may require sherpa-onnx v1.13+"
                warn "You have v${SHERPA_ONNX_VERSION}. Proceeding anyway..."
            fi
            return
        fi
    done
}

# ============================================================================
# Fetch model list from GitHub API
# ============================================================================

fetch_models() {
    local tag=$1
    local filter=${2:-""}
    
    info "Fetching model list from GitHub (tag: $tag)..."
    
    # Use GitHub API to get release assets
    local url="${GITHUB_API}/${tag}"
    local json
    
    json=$(curl -sL -H "Accept: application/vnd.github+json" "$url" 2>/dev/null)
    
    if [[ -z "$json" ]] || echo "$json" | grep -q "rate limit"; then
        warn "GitHub API rate limited or unavailable, using fallback method"
        # Fallback: scrape the release page HTML
        local html
        html=$(curl -sL "https://github.com/k2-fsa/sherpa-onnx/releases/tag/${tag}" 2>/dev/null)
        echo "$html" | grep -oP "(?<=/download/${tag}/)[^\"]+\.tar\.bz2" | sed 's/\.tar\.bz2$//' | sort -u
        return
    fi
    
    # Parse JSON to extract asset names
    echo "$json" | grep -oP '"name":\s*"\K[^"]+\.tar\.bz2' | sed 's/\.tar\.bz2$//' | sort -u
}

list_asr_models() {
    local filter=${1:-""}
    
    echo ""
    echo -e "${CYAN}Available ASR Models (from asr-models release):${NC}"
    echo "================================================"
    echo ""
    
    local models
    models=$(fetch_models "asr-models")
    
    if [[ -n "$filter" ]]; then
        echo "$models" | grep -i "$filter" || echo "No models matching '$filter'"
    else
        echo "$models" | column -c 100 2>/dev/null || echo "$models"
    fi
    
    echo ""
    echo -e "Total: $(echo "$models" | wc -l) models"
    echo ""
    echo -e "Common choices:"
    echo "  sherpa-onnx-whisper-large-v3       (best accuracy, large)"
    echo "  sherpa-onnx-whisper-turbo          (fast + accurate)"
    echo "  sherpa-onnx-whisper-small          (balanced)"
    echo "  sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17  (multilingual)"
    echo "  sherpa-onnx-moonshine-base-en-int8 (ultra fast, English)"
    echo ""
}

list_tts_models() {
    local filter=${1:-""}
    
    echo ""
    echo -e "${CYAN}Available TTS Models (from tts-models release):${NC}"
    echo "================================================"
    echo ""
    
    local models
    models=$(fetch_models "tts-models")
    
    if [[ -n "$filter" ]]; then
        echo "$models" | grep -i "$filter" || echo "No models matching '$filter'"
    else
        echo "$models" | column -c 100 2>/dev/null || echo "$models"
    fi
    
    echo ""
    echo -e "Total: $(echo "$models" | wc -l) models"
    echo ""
    echo -e "Common choices:"
    echo "  vits-piper-de_DE-thorsten-high     (German, high quality)"
    echo "  vits-piper-en_US-lessac-medium     (English US)"
    echo "  vits-piper-en_GB-alba-medium       (English UK)"
    echo "  vits-coqui-en-ljspeech             (English, classic)"
    echo "  kokoro-multi-lang-v1_0             (multi-language)"
    echo ""
}

# ============================================================================
# Download functions
# ============================================================================

download_model() {
    local tag=$1
    local model_name=$2
    local output_name=$3
    
    # Remove .tar.bz2 if user included it
    model_name="${model_name%.tar.bz2}"
    
    # Check version compatibility
    check_model_compatibility "$model_name"
    
    local url="${GITHUB_BASE}/${tag}/${model_name}.tar.bz2"
    local tarball="${model_name}.tar.bz2"
    
    info "sherpa-onnx version: ${SHERPA_ONNX_VERSION}"
    info "Downloading: $model_name"
    info "URL: $url"
    
    mkdir -p "$MODELS_DIR"
    cd "$MODELS_DIR"
    
    # Download
    if [[ -f "$tarball" ]]; then
        warn "Tarball already exists, skipping download"
    else
        if ! curl -L -f --progress-bar -o "$tarball" "$url"; then
            rm -f "$tarball"
            error "Failed to download. Model may not exist: $model_name"
        fi
    fi
    
    # Verify it's a valid bzip2 file
    if ! file "$tarball" | grep -q "bzip2"; then
        rm -f "$tarball"
        error "Downloaded file is not a valid bzip2 archive"
    fi
    
    info "Extracting $tarball..."
    tar -xjf "$tarball"
    
    # Move to output directory
    rm -rf "$output_name"
    mv "$model_name" "$output_name"
    echo "$model_name" > "$output_name/.model_name"
    rm -f "$tarball"
    
    cd - > /dev/null
    success "Downloaded $model_name to $MODELS_DIR/$output_name"
}

download_asr() {
    local model="${1:-$DEFAULT_ASR}"
    download_model "asr-models" "$model" "asr"
}

download_tts() {
    local model="${1:-$DEFAULT_TTS}"
    download_model "tts-models" "$model" "tts"
}

# ============================================================================
# Help and main
# ============================================================================

show_help() {
    echo "Wyoming Sherpa-ONNX Model Downloader"
    echo "Sherpa-ONNX Version: ${SHERPA_ONNX_VERSION}"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  all [asr] [tts]          Download ASR and TTS models (defaults if omitted)"
    echo "  asr <model-name>         Download specific ASR model"
    echo "  tts <model-name>         Download specific TTS model"
    echo "  --list-asr [filter]      List available ASR models (optionally filtered)"
    echo "  --list-tts [filter]      List available TTS models (optionally filtered)"
    echo "  --version                Show sherpa-onnx version"
    echo "  --help                   Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  MODELS_DIR               Directory to store models (default: ./models)"
    echo ""
    echo "Examples:"
    echo "  $0 all                                    # Download defaults"
    echo "  $0 asr sherpa-onnx-whisper-large-v3       # Download Whisper large-v3"
    echo "  $0 asr sherpa-onnx-whisper-turbo          # Download Whisper turbo"
    echo "  $0 tts vits-piper-en_US-lessac-medium     # Download English Piper"
    echo "  $0 --list-asr whisper                     # List Whisper models"
    echo "  $0 --list-tts piper-de                    # List German Piper models"
    echo ""
    echo -e "${YELLOW}Compatibility:${NC}"
    echo "  Models must be compatible with sherpa-onnx v${SHERPA_ONNX_VERSION}"
    echo "  SenseVoice models require v1.13+ for newer releases"
    echo ""
    echo "Model sources:"
    echo "  ASR: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models"
    echo "  TTS: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models"
    echo ""
}

# Main
case "${1:-}" in
    all)
        download_asr "${2:-$DEFAULT_ASR}"
        download_tts "${3:-$DEFAULT_TTS}"
        echo ""
        success "All models downloaded to $MODELS_DIR"
        echo "  ASR: $MODELS_DIR/asr"
        echo "  TTS: $MODELS_DIR/tts"
        ;;
    asr)
        [[ -z "${2:-}" ]] && error "Model name required. Use --list-asr to see available models."
        download_asr "$2"
        ;;
    tts)
        [[ -z "${2:-}" ]] && error "Model name required. Use --list-tts to see available models."
        download_tts "$2"
        ;;
    --list-asr|-la)
        list_asr_models "${2:-}"
        ;;
    --list-tts|-lt)
        list_tts_models "${2:-}"
        ;;
    --help|-h)
        show_help
        ;;
    --version|-v)
        echo "sherpa-onnx version: ${SHERPA_ONNX_VERSION}"
        ;;
    "")
        show_help
        ;;
    *)
        error "Unknown command: $1. Use --help for usage."
        ;;
esac
