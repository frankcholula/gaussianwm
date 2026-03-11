FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="8.0;8.6"

# System dependencies and Python 3.10 setup
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
    libglib2.0-0 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && pip install uv

# CUDA environment for extension compilation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCHS}
ENV MAX_JOBS=4

WORKDIR /app

# Copy dependency files and strip unused packages (swap opencv for headless variant)
COPY pyproject.toml uv.lock ./

RUN sed -i \
    -e '/pyrealsense2/d' \
    -e '/trimesh/d' \
    -e '/imageio-ffmpeg/d' \
    -e '/opencv-contrib-python/d' \
    -e '/open3d/d' \
    -e '/ipdb/d' \
    -e '/atomics/d' \
    -e '/gradio/d' \
    -e '/viser/d' \
    -e '/peft/d' \
    -e '/huggingface_hub/d' \
    -e '/transformers/d' \
    -e 's/opencv-python/opencv-python-headless/' \
    pyproject.toml

# Install Python dependencies including Splatt3r submodule dependencies
RUN uv sync --no-install-project \
    && uv pip install scikit-learn scikit-image scipy roma gitpython

ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

# Copy submodule source before CUDA compilation
COPY third_party/ ./third_party/

# Install CUDA extension packages (require torch already installed)
RUN uv pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified --no-build-isolation \
    && uv pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation

# Compile CroCo CUDA kernels
WORKDIR /app/third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope
RUN python setup.py build_ext --inplace
WORKDIR /app

# Copy project source and install
COPY gaussianwm/ ./gaussianwm/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY CLAUDE.md README.md ./

RUN uv pip install -e . --no-build-isolation

COPY tests/test_install.py ./tests/test_install.py

# Environment setup
ENV GWM_PATH=/app
ENV PYTHONPATH="/app:/app/gaussianwm:/app/third_party/splatt3r:/app/third_party/splatt3r/src/pixelsplat_src:/app/third_party/splatt3r/src/mast3r_src:/app/third_party/splatt3r/src/mast3r_src/dust3r"

RUN mkdir -p /app/data /app/logs \
    /app/third_party/splatt3r/checkpoints/splatt3r_v1.0

CMD ["/bin/bash"]
