# =============================================================================
# Gaussian World Model (GWM) Docker Image
# Python 3.10, PyTorch 2.5.1, CUDA 12.1
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="8.0;8.6"

# ---------------------------------------------------------------------------
# 1. System dependencies (minimal — no GUI/X11 libs)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    build-essential \
    git \
    curl \
    ninja-build \
    libgomp1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# CUDA env for compiling extensions
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCHS}
ENV MAX_JOBS=4

# ---------------------------------------------------------------------------
# 2. Install uv
# ---------------------------------------------------------------------------
RUN pip install uv

WORKDIR /app

# ---------------------------------------------------------------------------
# 3. Copy dependency files and strip unused packages
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./

# Remove 13 unused deps + swap opencv for headless variant
RUN sed -i \
    -e '/"pyrealsense2"/d' \
    -e '/"trimesh/d' \
    -e '/"imageio-ffmpeg/d' \
    -e '/"opencv-contrib-python"/d' \
    -e '/"open3d"/d' \
    -e '/"ipdb"/d' \
    -e '/"atomics"/d' \
    -e '/"gradio"/d' \
    -e '/"viser"/d' \
    -e '/"plyfile"/d' \
    -e '/"peft/d' \
    -e '/"huggingface_hub/d' \
    -e '/"transformers/d' \
    -e 's/"opencv-python"/"opencv-python-headless"/' \
    pyproject.toml

# ---------------------------------------------------------------------------
# 4. Install Python dependencies
# ---------------------------------------------------------------------------
RUN uv sync --no-install-project

# Put the venv on PATH
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

# ---------------------------------------------------------------------------
# 5. Copy submodule source (before CUDA compilation steps)
# ---------------------------------------------------------------------------
COPY third_party/ ./third_party/

# ---------------------------------------------------------------------------
# 6. Install CUDA extension packages (require torch already installed)
# ---------------------------------------------------------------------------
RUN uv pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified --no-build-isolation
RUN uv pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation

# ---------------------------------------------------------------------------
# 7. Compile CroCo CUDA kernels
# ---------------------------------------------------------------------------
WORKDIR /app/third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope
RUN python setup.py build_ext --inplace
WORKDIR /app

# ---------------------------------------------------------------------------
# 8. Copy project source and install
# ---------------------------------------------------------------------------
COPY gaussianwm/ ./gaussianwm/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY CLAUDE.md README.md ./

RUN uv pip install -e . --no-build-isolation

# ---------------------------------------------------------------------------
# 9. Copy verification test
# ---------------------------------------------------------------------------
COPY tests/test_install.py ./tests/test_install.py

# ---------------------------------------------------------------------------
# 10. Environment
# ---------------------------------------------------------------------------
ENV GWM_PATH=/app
ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN mkdir -p /app/data /app/logs \
    /app/third_party/splatt3r/checkpoints/splatt3r_v1.0

CMD ["python", "tests/test_install.py"]
