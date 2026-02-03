# Dockerfile for sherpa-onnx on NVIDIA Jetson Orin NX
# CUDA 12.6, cuDNN 9, Linux arm64 (JetPack 6.x / L4T 36.4.7)
# Based on: https://k2-fsa.github.io/sherpa/onnx/install/linux.html

# Use NVIDIA L4T JetPack image for Jetson Orin (JetPack 6.x)
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

LABEL maintainer="wyoming-sherpa-onnx"
LABEL description="Wyoming sherpa-onnx ASR/TTS for Jetson Orin NX (JetPack 6.x)"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV SHERPA_ONNX_VERSION=1.18.1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /build

# Clone and build sherpa-onnx with GPU support for Jetson Orin NX
# Using specific release for reproducible builds
RUN git clone --depth 1 --branch v1.12.23 https://github.com/k2-fsa/sherpa-onnx && \
    cd sherpa-onnx && \
    mkdir build && \
    cd build && \
    cmake \
        -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=${SHERPA_ONNX_VERSION} \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DSHERPA_ONNX_ENABLE_GPU=ON \
        .. && \
    make -j$(nproc)

# Copy sherpa-onnx binaries and libraries
RUN cp -r /build/sherpa-onnx/build/bin/* /usr/local/bin/ && \
    cp -r /build/sherpa-onnx/build/lib/* /usr/local/lib/ && \
    ldconfig

# Install Python sherpa-onnx from source build
RUN cd /build/sherpa-onnx && \
    pip3 install --no-cache-dir --break-system-packages .

# Clean up build directory to reduce image size
RUN rm -rf /build

# Set up application directory
WORKDIR /app

# Copy Wyoming handler
COPY pyproject.toml /app/
COPY wyoming_sherpa_onnx/ /app/wyoming_sherpa_onnx/

# Install Wyoming handler and dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    wyoming>=1.5.0 \
    numpy>=1.24.0 \
    soundfile>=0.12.0 \
    && pip3 install --no-cache-dir --break-system-packages -e /app

# Create directories for models
RUN mkdir -p /app/models/asr /app/models/tts

# Environment variables for runtime
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV USE_GPU=true
ENV ASR_MODEL_PATH=/app/models/asr
ENV TTS_MODEL_PATH=/app/models/tts
ENV TTS_SPEAKER_ID=0
ENV ASR_URI=tcp://0.0.0.0:10300
ENV TTS_URI=tcp://0.0.0.0:10400

# Expose both ASR and TTS ports
EXPOSE 10300 10400

# Run Wyoming server
CMD ["python3", "-m", "wyoming_sherpa_onnx"]
