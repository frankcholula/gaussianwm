# =============================================================================
# Gaussian World Model (GWM) Makefile
# Wraps Docker operations and training workflows
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration variables
# -----------------------------------------------------------------------------
IMAGE_NAME := gwm
IMAGE_TAG := latest
CONTAINER_NAME := gwm-container

# GPU configuration (override via: make run GPUS=0,1,2,3)
GPUS ?= all

# CUDA architectures for compilation (override via: make build CUDA_ARCHS="8.0;8.6;9.0")
CUDA_ARCHS ?= 8.0;8.6

# Volume mount paths
PWD := $(shell pwd)
DATA_DIR := $(PWD)/data
LOGS_DIR := $(PWD)/logs
CHECKPOINTS_DIR := $(PWD)/third_party/splatt3r/checkpoints

# Docker run flags
DOCKER_RUN_FLAGS := --rm --gpus $(GPUS) \
	--name $(CONTAINER_NAME) \
	--ipc=host \
	--shm-size=16g \
	-v $(PWD):/app \
	-v $(DATA_DIR):/app/data \
	-v $(LOGS_DIR):/app/logs \
	-v $(CHECKPOINTS_DIR):/app/third_party/splatt3r/checkpoints \
	-e CUDA_VISIBLE_DEVICES \
	-e WANDB_API_KEY \
	-e GWM_PATH=/app

# Default multi-GPU settings
NUM_GPUS ?= 4
MASTER_PORT ?= 12345

# -----------------------------------------------------------------------------
# Phony targets
# -----------------------------------------------------------------------------
.PHONY: help build run shell setup test clean \
	vae-single vae-multi dit dry-run demo \
	docker-clean

# -----------------------------------------------------------------------------
# Default target: show help
# -----------------------------------------------------------------------------
help:
	@echo "Gaussian World Model (GWM) - Makefile Targets"
	@echo "=============================================="
	@echo ""
	@echo "Docker Operations:"
	@echo "  make build         - Build Docker image (set CUDA_ARCHS='8.0;8.6;9.0' for custom GPU archs)"
	@echo "  make run           - Run container with GPU support and volume mounts"
	@echo "  make shell         - Open interactive bash shell in container"
	@echo "  make docker-clean  - Remove Docker image and dangling layers"
	@echo ""
	@echo "Local Setup:"
	@echo "  make setup         - Install dependencies locally (uv sync + CUDA extensions)"
	@echo "  make test          - Run installation verification test"
	@echo ""
	@echo "Training (runs in Docker container if image exists, else local):"
	@echo "  make vae-single    - Train VAE on single GPU"
	@echo "  make vae-multi     - Train VAE on multi-GPU (set NUM_GPUS=4, default=4)"
	@echo "  make dit           - Train Diffusion Transformer (multi-GPU)"
	@echo "  make dry-run       - VAE dry-run test"
	@echo "  make demo          - Run inference demo"
	@echo ""
	@echo "Utility:"
	@echo "  make clean         - Clean build artifacts and caches"
	@echo ""
	@echo "Environment Variables:"
	@echo "  GPUS=all           - GPU selection (default: all; override: 0,1,2,3)"
	@echo "  CUDA_ARCHS='8.0;8.6' - CUDA compute architectures for build"
	@echo "  NUM_GPUS=4         - Number of GPUs for multi-GPU training"
	@echo "  MASTER_PORT=12345  - PyTorch distributed master port"
	@echo "  WANDB_API_KEY      - Weights & Biases API key (auto-forwarded)"
	@echo ""
	@echo "Examples:"
	@echo "  make build CUDA_ARCHS='8.6'                    # RTX 3090 only"
	@echo "  make shell GPUS=0                              # Use GPU 0 only"
	@echo "  make vae-multi NUM_GPUS=2 GPUS=0,1            # 2-GPU training"
	@echo "  WANDB_API_KEY=xxx make vae-single             # With W&B logging"

# -----------------------------------------------------------------------------
# Docker: Build image
# -----------------------------------------------------------------------------
build:
	@echo "Building Docker image: $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "CUDA architectures: $(CUDA_ARCHS)"
	docker build \
		--build-arg CUDA_ARCHS="$(CUDA_ARCHS)" \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		.
	@echo ""
	@echo "Build complete. Run 'make test' to verify installation."

# -----------------------------------------------------------------------------
# Docker: Run container with volume mounts
# -----------------------------------------------------------------------------
run:
	@echo "Starting container: $(CONTAINER_NAME)"
	@echo "GPU selection: $(GPUS)"
	@mkdir -p $(DATA_DIR) $(LOGS_DIR) $(CHECKPOINTS_DIR)
	docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME):$(IMAGE_TAG)

# -----------------------------------------------------------------------------
# Docker: Interactive shell
# -----------------------------------------------------------------------------
shell:
	@echo "Opening interactive shell in container"
	@echo "GPU selection: $(GPUS)"
	@mkdir -p $(DATA_DIR) $(LOGS_DIR) $(CHECKPOINTS_DIR)
	docker run -it $(DOCKER_RUN_FLAGS) $(IMAGE_NAME):$(IMAGE_TAG) /bin/bash

# -----------------------------------------------------------------------------
# Docker: Clean images and dangling layers
# -----------------------------------------------------------------------------
docker-clean:
	@echo "Removing Docker image: $(IMAGE_NAME):$(IMAGE_TAG)"
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG)
	@echo "Removing dangling images..."
	-docker image prune -f
	@echo "Docker cleanup complete."

# -----------------------------------------------------------------------------
# Local: Setup environment
# -----------------------------------------------------------------------------
setup:
	@echo "Setting up local environment..."
	@echo "Step 1: Installing uv..."
	pip install uv
	@echo "Step 2: Running uv sync..."
	uv sync
	@echo "Step 3: Installing CUDA extension packages..."
	@echo "  - diff-gaussian-rasterization-modified"
	uv pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified --no-build-isolation
	@echo "  - pytorch3d"
	uv pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
	@echo ""
	@echo "Setup complete. Activate with: source .venv/bin/activate"
	@echo "Run 'make test' to verify installation."

# -----------------------------------------------------------------------------
# Test: Run installation verification
# -----------------------------------------------------------------------------
test:
	@if [ -f /.dockerenv ]; then \
		echo "Running test inside Docker container..."; \
		python tests/test_install.py; \
	elif docker images -q $(IMAGE_NAME):$(IMAGE_TAG) 2> /dev/null | grep -q .; then \
		echo "Running test in Docker container..."; \
		docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME):$(IMAGE_TAG) python tests/test_install.py; \
	else \
		echo "Running test locally (activate .venv first)..."; \
		python tests/test_install.py; \
	fi

# -----------------------------------------------------------------------------
# Training: VAE single GPU
# -----------------------------------------------------------------------------
vae-single:
	@echo "Starting VAE training (single GPU)..."
	@if [ -f /.dockerenv ]; then \
		python gaussianwm/train_vae.py --config-name=train_vae_single_gpu use_wandb=true; \
	elif docker images -q $(IMAGE_NAME):$(IMAGE_TAG) 2> /dev/null | grep -q .; then \
		docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME):$(IMAGE_TAG) \
			python gaussianwm/train_vae.py --config-name=train_vae_single_gpu use_wandb=true; \
	else \
		python gaussianwm/train_vae.py --config-name=train_vae_single_gpu use_wandb=true; \
	fi

# -----------------------------------------------------------------------------
# Training: VAE multi-GPU
# -----------------------------------------------------------------------------
vae-multi:
	@echo "Starting VAE training (multi-GPU: $(NUM_GPUS) GPUs)..."
	@if [ -f /.dockerenv ]; then \
		bash scripts/pretrain/vae.sh; \
	elif docker images -q $(IMAGE_NAME):$(IMAGE_TAG) 2> /dev/null | grep -q .; then \
		docker run $(DOCKER_RUN_FLAGS) \
			-e NUM_GPUS=$(NUM_GPUS) \
			-e MASTER_PORT=$(MASTER_PORT) \
			$(IMAGE_NAME):$(IMAGE_TAG) \
			bash scripts/pretrain/vae.sh; \
	else \
		bash scripts/pretrain/vae.sh; \
	fi

# -----------------------------------------------------------------------------
# Training: Diffusion Transformer
# -----------------------------------------------------------------------------
dit:
	@echo "Starting DiT training (multi-GPU)..."
	@if [ -f /.dockerenv ]; then \
		bash scripts/pretrain/dit.sh; \
	elif docker images -q $(IMAGE_NAME):$(IMAGE_TAG) 2> /dev/null | grep -q .; then \
		docker run $(DOCKER_RUN_FLAGS) \
			-e NUM_GPUS=$(NUM_GPUS) \
			-e MASTER_PORT=$(MASTER_PORT) \
			$(IMAGE_NAME):$(IMAGE_TAG) \
			bash scripts/pretrain/dit.sh; \
	else \
		bash scripts/pretrain/dit.sh; \
	fi

# -----------------------------------------------------------------------------
# Training: VAE dry-run test
# -----------------------------------------------------------------------------
dry-run:
	@echo "Running VAE dry-run test..."
	@if [ -f /.dockerenv ]; then \
		bash scripts/test_vae_dryrun.sh; \
	elif docker images -q $(IMAGE_NAME):$(IMAGE_TAG) 2> /dev/null | grep -q .; then \
		docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME):$(IMAGE_TAG) \
			bash scripts/test_vae_dryrun.sh; \
	else \
		bash scripts/test_vae_dryrun.sh; \
	fi

# -----------------------------------------------------------------------------
# Inference: Demo
# -----------------------------------------------------------------------------
demo:
	@echo "Running inference demo..."
	@if [ -f /.dockerenv ]; then \
		python gaussianwm/demo.py; \
	elif docker images -q $(IMAGE_NAME):$(IMAGE_TAG) 2> /dev/null | grep -q .; then \
		docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME):$(IMAGE_TAG) \
			python gaussianwm/demo.py; \
	else \
		python gaussianwm/demo.py; \
	fi

# -----------------------------------------------------------------------------
# Clean: Remove build artifacts and caches
# -----------------------------------------------------------------------------
clean:
	@echo "Cleaning build artifacts and caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "Clean complete."
